"""Read-only integration review for explicit learner stages and reusable ordered combination."""
import argparse
import importlib
import inspect
import json
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.architect.models import EdgeKind
from sciona.decomposed_amex_graph import build_decomposed_amex_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.promote_connectomics_execution import provider_ports, check_hashes
from scripts.review_amex_execution import audit, sha
from scripts.import_flavours_provider_drafts import connect

ROOT = Path(__file__).resolve().parents[1]
GENERIC = 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def boundary_ports(graph, interfaces):
    """Callable-only nodes expose function arguments to their caller, not users."""
    callable_only = {n.node_id for n in graph.nodes
        if any(e.source_id == n.node_id for e in graph.edges)
        and all(e.edge_kind == EdgeKind.CALLABLE_INJECTION for e in graph.edges if e.source_id == n.node_id)}
    require(not any(e.target_id in callable_only for e in graph.edges), 'Bound callable closures unsupported')
    connected = {(e.target_id, e.input_name) for e in graph.edges}
    consumed = {(e.source_id, e.output_name) for e in graph.edges}
    result = {'input': [], 'output': []}
    for node in graph.nodes:
        if node.node_id in callable_only:
            continue
        for direction, excluded in [('input', connected), ('output', consumed)]:
            result[direction].extend(p for p in interfaces[node.node_id][direction]
                                     if (node.node_id, p.name) not in excluded)
    for ports in result.values():
        require(len({p.name for p in ports}) == len(ports), 'Ambiguous root ports')
    return result


def review():
    reviews = ROOT/'docs/reviews'
    original = json.loads((reviews/'competition_amex_semantic_review.json').read_text())
    require(audit() == original, 'Original source qualification changed')
    names = ['decomposed_amex_pipeline','decomposed_amex_graph']
    documents = {}
    for name in names:
        document=json.loads((reviews/(name+'.json')).read_text())
        require(document['passed'] is True and document['approved'] is False and document['catalog_mutations']==0,'Invalid qualification status')
        check_hashes(ROOT,document['implementation_sha256'])
        documents[name]=document
    parity=documents[names[0]]
    require(parity['complete_reference_models']==parity['complete_candidate_models']==25,'Complete learner parity required')
    require(all(parity[k] is True for k in ('exact_branch_probabilities','exact_final_output','input_and_torch_rng_preserved')),'Full recipe parity missing')
    full=documents[names[1]]
    graph=build_decomposed_amex_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    require((len(nodes),len(edges))==(10,16) and full['serialized_graph_sha256']==digest,'Qualified graph changed')
    expected=dict(actual_runner_nodes=10,edges=16,models=25,dart_models=15,rounds_per_dart_model=4500,neural_models=10,epochs_per_neural_model=10)
    require(all(full[k]==v for k,v in expected.items()),'Complete graph learner controls missing')
    require(all(full[k] is True for k in ('all_nodes_uncached','exact_source_blend','strict_json','input_and_torch_rng_preserved','exact_reference_output')),'Graph execution invariant missing')
    records=full['training_records']
    require(set(records)=={'reference','graph'} and records['reference']==records['graph'] and len(records['graph'])==15,'Source training trace differs')
    require(all(r['evaluated_rounds']==4500 and type(r['retained_iterations']) is int and 1<=r['retained_iterations']<=4500 for r in records['graph']),'Incomplete boosting iteration trace')
    wanted = {n.matched_primitive for n in graph.nodes}
    specs = {}
    for repo in ('sciona-atoms', 'sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in wanted:
                require(runtime not in specs, 'Ambiguous provider identity')
                specs[runtime] = spec
    require(set(specs) == wanted and len(specs) == 10, 'Missing runtime provider')
    records, interfaces = [], {}
    for runtime, spec in sorted(specs.items()):
        function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        require(Path(inspect.getfile(function)).resolve() == spec.file_path.resolve(), 'Provider implementation differs')
        expected = (full['generic_provider_sha256'] if runtime == GENERIC else
                    original['provider_sha256'] if runtime.endswith('.amex_prepare') else
                    full['provider_sha256'])
        require(sha(spec.file_path) == expected, 'Provider source changed')
        if runtime == GENERIC:
            require(all(d['generic_provider_sha256'] == expected for d in documents.values()), 'Generic evidence differs')
        bound = [n for n in graph.nodes if n.matched_primitive == runtime]
        for node in bound:
            interfaces[node.node_id] = provider_ports(node, function)
        records.append(dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
            version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=expected,
            signature=str(inspect.signature(function)), bound_nodes=[n.node_id for n in bound],
            classification='generic_operation' if runtime == GENERIC else 'source_recipe_adapter'))
    boundary = boundary_ports(graph, interfaces)
    require([p.name for p in boundary['input']] == ['payload'] and
            [p.name for p in boundary['output']] == ['result'], 'Unexpected root interface')
    return dict(format='amex-decomposed-review.v1', review_source='automated',
        proposed_tier=3, approved=False, catalog_mutations=0, graph_digest=digest,
        graph_nodes=10, graph_edges=16, providers=records, root_inputs=['payload'], root_outputs=['result'],
        source_stage_mapping=original['intake_stage_mapping'],
        evidence_sha256={name+'.json': sha(reviews/(name+'.json')) for name in names+
                        ['competition_amex_semantic_review']},
        auxiliary_sha256={name:sha(ROOT/name) for name in ['scripts/review_decomposed_amex.py',
            'tests/test_ordered_weighted_sum.py','tests/test_decomposed_amex_graph.py',
            'tests/test_decomposed_amex_provider_intake.py','tests/test_decomposed_amex_review.py']},
        limitations=[
            'Review only; provider eligibility, corrected original intake, promotion and served selection remain separate gates.',
            'Ordered row combination is domain independent; coefficients, branch order, preprocessing choices and learner controls remain recipe policy.',
            'No weight normalization, reassociation or replacement by averaging. Preserve literal source coefficient sum and floating arithmetic.',
            'Fifteen DART fits evaluate 4500 rounds each. Retained tree iterations may be lower when source learners produce empty trees; exact counts are compared with the unchanged reference.',
            'Ten neural fits retain all ten epochs, source effective learning rates, batch size and positive-metric checkpoint rules.',
            'Validation uses synthetic runtime roles and intercepts intermediate persistence in memory. No private input, checkpoint or identifying dataset metadata is committed.',
            'No historical GPU, RNG, trained-weight or competition-accuracy claim. Tier 1 human review and Tier 2 usage qualification are not claimed.'])



def inventory():
    result = review()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for record in result['providers']:
            for prefix, table, key in [('atom', 'atoms', 'atom_id'), ('artifact', 'artifacts', 'artifact_id')]:
                rows = db.execute(f'''SELECT a.{key}::text AS identity,a.status,a.is_publishable,
                    v.version_id::text,v.content_hash,v.trust_tier,
                    {prefix}_is_publishable(a.{key}) AS native_eligible
                    FROM {table} a JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s AND v.is_latest''', (record['catalog_fqdn'],)).fetchall()
                record[prefix+'_latest'] = rows
                record[prefix+'_exact_eligible'] = bool(len(rows) == 1 and
                    rows[0]['version_id'] == record['version_id'] and rows[0]['content_hash'] == record['content_hash']
                    and rows[0]['status'] == 'approved' and rows[0]['is_publishable']
                    and rows[0]['native_eligible'] and rows[0]['trust_tier'] in (1, 2, 3))
        result['original_state'] = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier FROM artifacts a
            JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest''',
            ('cdg.competition.solution.kaggle.amex_default_1st',)).fetchall()
    result['both_catalogs_exact_eligible'] = sum(r['atom_exact_eligible'] and r['artifact_exact_eligible']
                                                for r in result['providers'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', action='store_true')
    args = parser.parse_args()
    result = inventory() if args.catalog else review()
    name = 'amex_decomposed_provider_inventory.json' if args.catalog else 'amex_decomposed_semantic_review.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('approved', 'catalog_mutations', 'graph_nodes', 'graph_edges')}
                     | {'eligible': result.get('both_catalogs_exact_eligible')}))

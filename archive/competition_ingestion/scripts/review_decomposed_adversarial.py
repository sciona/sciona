"""Read-only integration review for explicit reusable attack normalization."""
import argparse
import importlib
import inspect
import json
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.architect.models import EdgeKind
from sciona.decomposed_adversarial_graph import build_decomposed_adversarial_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.promote_connectomics_execution import provider_ports, check_hashes
from scripts.review_adversarial_execution import audit, sha
from scripts.import_flavours_provider_drafts import connect

ROOT = Path(__file__).resolve().parents[1]
GENERIC = 'sciona.atoms.numerical.statistics.row_scale.normalize_rows_by_scale'


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
    original = json.loads((reviews/'competition_adversarial_semantic_review.json').read_text())
    require(audit() == original, 'Original source qualification changed')
    names = ['adversarial_generic_row_scale_source_parity', 'decomposed_adversarial_lifecycle',
             'decomposed_adversarial_padded', 'decomposed_adversarial_graph']
    documents = {}
    for name in names:
        document = json.loads((reviews/(name+'.json')).read_text())
        require(document['result'] == 'passed' and document['approved'] is False
                and document['catalog_mutations'] == 0, 'Invalid qualification status')
        check_hashes(ROOT, document['sha256'])
        documents[name] = document
    component = documents[names[0]]
    require(component['checks']['source_tail_cases'] == 32 and
            component['checks']['source_nan_and_explicit_rejection_cases'] == 3,
            'Pinned source normalization coverage incomplete')
    require(len(component['complete_synthetic_gradient_schedules']) == 6 and
            all(c['exact'] for c in component['complete_synthetic_gradient_schedules']),
            'Complete source update schedules missing')
    require(component['test_sha256'] == sha(ROOT/'tests/test_row_scale.py'), 'Generic tests changed')
    full = documents[names[-1]]
    graph = build_decomposed_adversarial_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require((len(nodes), len(edges)) == (3, 2) and full['serialized_graph_sha256'] == digest,
            'Qualified graph changed')
    injection = [e for e in graph.edges if e.edge_kind == EdgeKind.CALLABLE_INJECTION]
    require(len(injection) == 1 and injection[0].source_id == 'normalize_rows'
            and injection[0].target_id == 'execute' and injection[0].input_name == 'normalizer',
            'Normalization must be injected into the recurrence')
    require(set(full['checks']) == {'non_targeted', 'targeted_large', 'targeted_small'}, 'Branch missing')
    for branch, models, steps, calls in [('non_targeted', 8, 10, 10),
                                        ('targeted_large', 5, 20, 40), ('targeted_small', 2, 40, 80)]:
        case = full['checks'][branch]
        require(all(case[k] == v for k, v in dict(actual_runner_nodes=3, full_models=models,
            complete_iterations=steps, exact_reference_update_checks=steps,
            registered_normalization_calls=calls, neural_batch_size=10, real_entries=1, padded_entries=9).items()),
            'Full graph execution coverage differs')
        require(all(case[k] is True for k in ('registered_callable_identity', 'only_real_entry_encoded',
            'decoded_png_matches_source_scaling', 'fixed_projection_bounds', 'input_and_rng_unchanged')),
            'Execution invariant failed')
    wanted = {n.matched_primitive for n in graph.nodes}
    specs = {}
    for repo in ('sciona-atoms', 'sciona-atoms-dl'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in wanted:
                require(runtime not in specs, 'Ambiguous provider identity')
                specs[runtime] = spec
    require(set(specs) == wanted and len(specs) == 3, 'Missing runtime provider')
    records, interfaces = [], {}
    for runtime, spec in sorted(specs.items()):
        function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        require(Path(inspect.getfile(function)).resolve() == spec.file_path.resolve(), 'Provider implementation differs')
        expected = (full['generic_provider_sha256'] if runtime == GENERIC else
                    original['provider_sha256'] if runtime.endswith('.adversarial_prepare') else
                    full['execution_provider_sha256'])
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
    return dict(format='adversarial-decomposed-review.v1', review_source='automated',
        proposed_tier=3, approved=False, catalog_mutations=0, graph_digest=digest,
        graph_nodes=3, graph_edges=2, providers=records, root_inputs=['payload'], root_outputs=['result'],
        source_stage_mapping=original['intake_stage_mapping'],
        evidence_sha256={name+'.json': sha(reviews/(name+'.json')) for name in names+
                        ['competition_adversarial_semantic_review']},
        auxiliary_sha256={name: sha(ROOT/name) for name in ['scripts/review_decomposed_adversarial.py',
            'tests/test_injected_adversarial_update.py', 'tests/test_runner_callable_injection.py',
            'tests/test_decomposed_adversarial_review.py']},
        limitations=[
            'Review only; provider eligibility, corrected original intake, promotion and served selection remain separate gates.',
            'The generic row operation is domain independent. Model choices, image layout, normalization policy and recurrence are source recipe semantics.',
            'Callable dependency is invoked ten, forty or eighty times in the qualified schedules; it is not a one-time preprocessing operation.',
            'Caller-supplied normalization is a trusted graph dependency; numerical fidelity requires the reviewed provider binding.',
            'Synthetic states and exact CPU-update agreement do not establish pretrained quality or historical TensorFlow binary equivalence.',
            'Undefined normalization rejects without a stabilizer. Source PNG stretching does not preserve a pixel epsilon guarantee.',
            'Validation intercepts intermediate persistence in memory. Only synthetic aggregate evidence is committed.',
            'Tier 1 human review and Tier 2 usage qualification are not claimed.'])


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
            ('cdg.competition.solution.kaggle.adversarial_attacks_1st',)).fetchall()
    result['both_catalogs_exact_eligible'] = sum(r['atom_exact_eligible'] and r['artifact_exact_eligible']
                                                for r in result['providers'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', action='store_true')
    args = parser.parse_args()
    result = inventory() if args.catalog else review()
    name = 'adversarial_decomposed_provider_inventory.json' if args.catalog else 'adversarial_decomposed_semantic_review.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('approved', 'catalog_mutations', 'graph_nodes', 'graph_edges')}
                     | {'eligible': result.get('both_catalogs_exact_eligible')}))

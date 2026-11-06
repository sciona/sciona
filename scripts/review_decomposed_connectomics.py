"""Review reusable filtering integration and inspect exact live provider eligibility.

No catalog writes. Prior source review remains an immutable prerequisite.
"""
import argparse
import importlib
import inspect
import json
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_connectomics_graph import build_decomposed_connectomics_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.import_flavours_provider_drafts import connect
from scripts.promote_connectomics_execution import provider_ports, boundary_ports, check_hashes
from scripts.review_connectomics_execution import ROOT, audit, sha

GENERIC = 'sciona.atoms.numerical.spatial.circular_filter.ordered_circular_filter'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review():
    reviews = ROOT / 'docs/reviews'
    original = json.loads((reviews/'competition_connectomics_semantic_review.json').read_text())
    require(original == audit(), 'Prior source semantic evidence drift')
    full_path = reviews/'connectomics_decomposed_graph_execution.json'
    full = json.loads(full_path.read_text())
    graph = build_decomposed_connectomics_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require((len(nodes), len(edges)) == (7, 22), 'Decomposition topology changed')
    require(full['status'] == 'passed' and full['serialized_graph_sha256'] == digest,
            'Full execution evidence missing or stale')
    require(full['actual_runner_nodes'] == 7 and all(full[k] is True for k in (
        'all_nodes_uncached', 'strict_json_output', 'graph_codec_roundtrip',
        'independent_directivity_exact', 'caller_input_preserved',
        'degenerate_covariance_rejected', 'tampered_source_rejected')), 'Execution gates incomplete')
    require(len(full['cases']) == 4 and {
        (c['mode'], c['directivity'], c['pca_fits'], c['exact_source_match']) for c in full['cases']
    } == {(m, d, 240 if m == 'simple' else 480, True)
          for m in ('simple', 'tuned') for d in (False, True)}, 'Full source sweep coverage incomplete')
    required_files = {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('*connectomics*.py')}
    required_files.add('scripts/validate_decomposed_connectomics_graph.py')
    require(set(full['code_sha256']) == required_files, 'Implementation evidence incomplete')
    check_hashes(ROOT, full['code_sha256'])
    component_path = reviews/'connectomics_generic_circular_filter_source_parity.json'
    component = json.loads(component_path.read_text())
    require(component['passed'] is True and component['component_cases'] == 64
            and component['caller_inputs_preserved'] is True, 'Generic filter qualification incomplete')
    for key, name in [('validator_sha256', 'scripts/validate_connectomics_circular_filter.py'),
                      ('test_sha256', 'tests/test_ordered_circular_filter.py'),
                      ('source_loader_sha256', 'sciona/connectomics_source.py'),
                      ('source_pins_sha256', 'docs/reviews/competition_connectomics_source_pins.json')]:
        require(component[key] == sha(ROOT/name), 'Generic evidence drift: ' + key)
    require(len(component['sweeps']) == 2 and {c['mode'] for c in component['sweeps']} == {'simple', 'tuned'} and all(
        c['exact_scores'] is True and c['exact_optional_directivity_blend'] is True
        and c['pca_fits'] == (240 if c['mode'] == 'simple' else 480)
        and c['filter_calls'] == {f: 120 if c['mode'] == 'tuned' or f in ('f1', 'f2') else 0
                                 for f in ('f1', 'f2', 'f3', 'f4')}
        for c in component['sweeps']), 'Generic source sweep evidence incomplete')
    wanted = {n.matched_primitive for n in graph.nodes}
    specs = {}
    for repo in ('sciona-atoms', 'sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module + '.' + spec.source_symbol
            if runtime in wanted:
                require(runtime not in specs, 'Ambiguous provider')
                specs[runtime] = spec
    require(set(specs) == wanted and len(specs) == 4, 'Missing graph provider')
    interfaces = {}
    records = []
    for runtime, spec in sorted(specs.items()):
        function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        require(Path(inspect.getsourcefile(inspect.unwrap(function))).resolve() == spec.file_path.resolve(),
                'Runtime implementation differs')
        expected_hash = (full['circular_provider_sha256'] if runtime == GENERIC else
                         original['provider_sha256'] if runtime.endswith('.connectomics_prepare') else full['provider_sha256'])
        require(sha(spec.file_path) == expected_hash, 'Qualified provider source drift')
        if runtime == GENERIC:
            require(component['provider_sha256'] == expected_hash, 'Generic component source differs')
        bound = [n for n in graph.nodes if n.matched_primitive == runtime]
        for node in bound:
            interfaces[node.node_id] = provider_ports(node, function)
        records.append(dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
            version_id=str(spec.version_id), content_hash=spec.content_hash,
            provider_sha256=expected_hash, signature=str(inspect.signature(function)),
            bound_nodes=[n.node_id for n in bound],
            classification='generic_operation' if runtime == GENERIC else 'source_recipe_adapter'))
    boundary = boundary_ports(graph, interfaces)
    require([p.name for p in boundary['input']] == ['payload'] and
            [p.name for p in boundary['output']] == ['result'], 'Graph root interface changed')
    return dict(format='connectomics-decomposed-review.v1', review_source='automated',
        proposed_tier=3, approved=False, catalog_mutations=0, graph_digest=digest,
        graph_nodes=7, graph_edges=22, providers=records,
        source_stage_mapping=original['intake_stage_mapping'],
        evidence_sha256={p.name: sha(p) for p in (full_path, component_path,
            reviews/'competition_connectomics_semantic_review.json')},
        auxiliary_sha256={p: sha(ROOT/p) for p in ('scripts/review_decomposed_connectomics.py',
            'tests/test_decomposed_connectomics.py', 'tests/test_ordered_circular_filter.py',
            'tests/test_decomposed_connectomics_review.py')},
        limitations=[
            'This integration review does not approve or stage catalog records.',
            'The generic filter is domain independent; tap choice, time-axis policy and the remaining sweep are source recipe semantics.',
            'All four filters run in both modes; only f1/f2 are consumed by the simple sweep.',
            'Prefiltered recipe inputs are trusted graph bindings. Shape and finiteness validation alone does not prove correct tap computation.',
            'Historical PCA, source namespace callbacks and matcher runtime adapters still require explicit dependency review.',
            'Synthetic exact-source agreement does not establish historical accuracy or domain validity.'])


def inventory():
    report = review()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for record in report['providers']:
            for prefix, table, key in [('atom', 'atoms', 'atom_id'), ('artifact', 'artifacts', 'artifact_id')]:
                rows = db.execute(f'''SELECT a.{key}::text AS identity,a.status,a.is_publishable,
                    v.version_id::text,v.content_hash,v.trust_tier,
                    {prefix}_is_publishable(a.{key}) AS native_eligible
                    FROM {table} a JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s AND v.is_latest''', (record['catalog_fqdn'],)).fetchall()
                record[prefix+'_latest'] = rows
                record[prefix+'_exact_eligible'] = bool(len(rows) == 1
                    and rows[0]['version_id'] == record['version_id']
                    and rows[0]['content_hash'] == record['content_hash']
                    and rows[0]['status'] == 'approved' and rows[0]['is_publishable']
                    and rows[0]['native_eligible'] and rows[0]['trust_tier'] in (1, 2, 3))
        report['original_state'] = db.execute('''SELECT a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier FROM artifacts a
            JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest''',
            ('cdg.competition.solution.kaggle.connectomics_1st',)).fetchall()
    report['both_catalogs_exact_eligible'] = sum(
        r['atom_exact_eligible'] and r['artifact_exact_eligible'] for r in report['providers'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', action='store_true')
    args = parser.parse_args()
    result = inventory() if args.catalog else review()
    name = 'connectomics_decomposed_provider_inventory.json' if args.catalog else 'connectomics_decomposed_semantic_review.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('approved', 'catalog_mutations', 'graph_nodes', 'graph_edges')}
                     | {'providers': len(result['providers']), 'eligible': result.get('both_catalogs_exact_eligible')}))

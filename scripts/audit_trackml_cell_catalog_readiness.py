"""Read-only exact catalog binding audit for the decomposed original TrackML scope."""
import importlib
import inspect
import json
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_trackml_cell_graph import build_decomposed_trackml_cell_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.audit_decomposed_trackml_readiness import ROOT, audit, sha
from scripts.import_flavours_provider_drafts import connect

# Numeric providers are separate from the recipe operations that use them.
STAGES = {
    'detector_geometry_autodiscovery': ['kmeans', 'mean_shift'],
    'coordinate_rescaling_for_knn': ['cylinder', 'plane'],
    'per_layer_knn_search': ['search_fit', 'search_k', 'search_radius'],
    'circle_from_three_points': ['circle'],
    'helix_pitch_least_squares': ['pitch_two', 'pitch_three'],
    'helix_cylinder_intersection': ['cylinder_intersection'],
    'helix_cap_intersection': ['plane_intersection'],
    'perturbative_helix_correction': ['cap_correction', 'cylinder_correction', 'grid_fit', 'grid_evaluate'],
    'helix_nearest_point_distance': ['nearest'],
    'bayesian_neighbor_evaluation': ['decision', 'displacement', 'uncertainty'],
    'greedy_track_commit': ['commit'],
}


def review():
    baseline = audit()
    if baseline['implementation_drift']:
        raise ValueError('Retained source implementation drift')
    if set(STAGES) != {s['stage_id'] for s in baseline['source_stages']}:
        raise ValueError('Original stage scope changed')
    graph = build_decomposed_trackml_cell_graph()
    evidence_path = ROOT/'docs/reviews/decomposed_trackml_cell_graph_execution.json'
    evidence = json.loads(evidence_path.read_text())
    if not evidence['passed'] or encode_execution_graph(graph)[0] != evidence['graph_sha256']:
        raise ValueError('Qualified graph changed')
    for relative, digest in evidence['implementation_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Qualified graph implementation changed')
    wanted = {n.matched_primitive for n in graph.nodes}
    specs = {}
    for name in ('sciona-atoms-physics', 'sciona-atoms-ml'):
        directory = ROOT.parent/name
        for spec in _parse_registered_atoms(repo=ProviderRepo(name, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in wanted:
                if runtime in specs:
                    raise ValueError('Ambiguous registered provider')
                specs[runtime] = spec
    if set(specs) != wanted:
        raise ValueError('Graph provider absent from source inventory')
    bindings = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for node in graph.nodes:
            runtime = node.matched_primitive
            spec = specs[runtime]
            module, symbol = runtime.rsplit('.', 1)
            function = getattr(importlib.import_module(module), symbol)
            if sha(inspect.getfile(inspect.unwrap(function))) != evidence['all_provider_sha256'][runtime]:
                raise ValueError('Executed provider source changed')
            catalogs = {}
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                rows = db.execute(f'''SELECT a.{key}::text AS identity,a.status,a.is_publishable,
                    v.version_id::text,v.content_hash,v.trust_tier,v.is_latest
                    FROM {prefix}s a LEFT JOIN {prefix}_versions v ON v.{key}=a.{key} AND v.is_latest
                    WHERE a.fqdn=%s''', (spec.fqdn,)).fetchall()
                if len(rows) > 1:
                    raise ValueError('Ambiguous latest catalog binding')
                row = rows[0] if rows else None
                exact = bool(row and row['version_id'] == str(spec.version_id) and row['content_hash'] == spec.content_hash)
                served = bool(row and db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s', (row['identity'],)).fetchone())
                ready = bool(exact and row['status'] == 'approved' and row['is_publishable'] and row['trust_tier'] == 3 and served)
                catalogs[prefix] = dict(record=row, exact_source_version=exact, served=served, ready=ready)
            bindings.append(dict(node_id=node.node_id, runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
                content_hash=spec.content_hash, catalogs=catalogs, ready=all(c['ready'] for c in catalogs.values())))
    by_node = {b['node_id']: b for b in bindings}
    return dict(format='trackml-cell-catalog-readiness.v1', read_only=True, catalog_mutations=0,
        original_approved=baseline['original']['status'] == 'approved',
        graph_sha256=evidence['graph_sha256'], execution_evidence_sha256=sha(evidence_path),
        auditor_sha256=sha(__file__), bindings=bindings,
        ready_nodes=sum(b['ready'] for b in bindings), total_nodes=len(bindings),
        unresolved_nodes=[b['node_id'] for b in bindings if not b['ready']],
        original_stage_coverage=[dict(stage_id=s, numeric_nodes=nodes,
            numeric_bindings_ready=all(by_node[n]['ready'] for n in nodes)) for s,nodes in STAGES.items()],
        limitations=[
            'Exact source-version approval and served identity only; interface, dependency, recipe semantics and original graph publication gates remain separate.',
            'Stage mapping identifies reusable numerical operations, not proof of complete original stage semantics.',
            'The tracking adapter retains detector topology, candidate lifecycle, retry, ranking, map dispatch and commit policy.',
            'Learning calibration maps is not claimed: the original correction stage applies caller-supplied maps.',
            'Synthetic execution does not establish historical competition accuracy.'],
        next_actions=[
            'Qualify and stage any unresolved graph provider with exact interfaces and full recipe dependency closure.',
            'Review original stage semantics against retained source, including degenerate geometry and source-specific retry policy.',
            'Stage a corrected original graph only after all exact bindings and source coverage gates pass.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_cell_catalog_readiness.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('original_approved','ready_nodes','total_nodes','unresolved_nodes')}))

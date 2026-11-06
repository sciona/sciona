"""Read-only intake inventory for direct DSB bindings and qualified generic calls.

This is an explicit provider inventory, not a complete transitive dependency
closure: model constructors, local imports and stateful callbacks need review.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from sciona.dsb_decomposed_evaluation_graph import build_dsb_evaluation_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import ROOT, connect

GENERIC = {
    'sciona.atoms.numerical.spatial.coordinate_scaling.scale_coordinate_rows': 'dsb_generic_coordinate_scaling_source_parity.json',
    'sciona.atoms.dl.spatial.centered_window.centered_window': 'dsb_generic_center_window_source_parity.json',
    'sciona.atoms.numerical.statistics.threshold_sampling.repeat_rows_by_threshold': 'dsb_generic_sampling_source_parity.json',
    'sciona.atoms.numerical.statistics.threshold_sampling.sequential_softmax_indices': 'dsb_generic_sampling_source_parity.json',
    'sciona.atoms.numerical.spatial.volume_tiles.split_volume_tiles': 'dsb_generic_tiles_source_parity.json',
    'sciona.atoms.numerical.spatial.volume_tiles.combine_volume_tiles': 'dsb_generic_tiles_source_parity.json',
    'sciona.atoms.numerical.spatial.resampling.resample_regular_volume': 'dsb_generic_resampling_source_parity.json',
    'sciona.atoms.dl.probability_aggregation.noisy_or_with_baseline': 'dsb_generic_noisy_or_source_parity.json',
    'sciona.atoms.dl.loss.thresholded_log.thresholded_log_penalty': 'dsb_generic_log_penalty_source_parity.json',
    'sciona.atoms.dl.loss.ranked_selection.highest_scoring_indices': 'dsb_generic_ranked_selection_source_parity.json',
}


SPATIAL_RECIPES = {'sciona.atoms.dl.detection.dsb_spatial_recipes.' + name for name in
                   ('crop_detection', 'augment_detection', 'augment_classifier',
                    'crop_classifier_proposal', 'assign_anchor_labels')}


LIFECYCLE_STEPS = {'sciona.atoms.dl.detection.dsb_lifecycle_steps.' + name for name in
                   ('validate_detector', 'validate_classifier', 'export_model_checkpoint')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(offline=False):
    graph = build_dsb_evaluation_graph()
    digest, _, _ = encode_execution_graph(graph)
    evidence_path = ROOT / 'docs/reviews/dsb_lifecycle_contract_graph_source_parity.json'
    evidence = json.loads(evidence_path.read_text())
    if not evidence['passed'] or evidence['execution_graph_sha256'] != digest:
        raise ValueError('Current graph lacks matching full source evidence')
    for key, base in [('implementation_sha256', ROOT), ('provider_sha256', ROOT.parent)]:
        for name, expected in evidence[key].items():
            if sha(base / name) != expected:
                raise ValueError('Execution evidence source drift: ' + name)
    direct = {node.matched_primitive for node in graph.nodes}
    wanted = direct | GENERIC.keys() | SPATIAL_RECIPES | LIFECYCLE_STEPS
    specs = {}
    for repo_name in ('sciona-atoms', 'sciona-atoms-dl'):
        directory = ROOT.parent / repo_name
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo_name, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module + '.' + spec.source_symbol
            if runtime in wanted:
                if runtime in specs:
                    raise ValueError('Ambiguous provider: ' + runtime)
                specs[runtime] = spec
    if set(specs) != wanted:
        raise ValueError('Missing provider sources: ' + str(sorted(wanted - specs.keys())))
    records = []
    with (nullcontext(None) if offline else connect()) as db:
        if db is not None: db.execute('SET TRANSACTION READ ONLY')
        for runtime in sorted(wanted):
            spec = specs[runtime]
            function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
            if Path(inspect.getsourcefile(inspect.unwrap(function))).resolve() != spec.file_path.resolve():
                raise ValueError('Runtime provider source differs')
            record = dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
                          version_id=str(spec.version_id), content_hash=spec.content_hash,
                          provider_sha256=sha(spec.file_path), signature=str(inspect.signature(function)),
                          directly_bound_nodes=[n.node_id for n in graph.nodes if n.matched_primitive == runtime],
                          classification='generic_operation' if runtime in GENERIC else 'dsb_recipe_adapter')
            if runtime in GENERIC:
                path = ROOT/'docs/reviews'/GENERIC[runtime]
                report = json.loads(path.read_text())
                if not report['passed'] or report['provider_sha256'] != record['provider_sha256']:
                    raise ValueError('Generic provider qualification drift: ' + runtime)
                record['component_evidence'] = dict(report=GENERIC[runtime], sha256=sha(path))
            for prefix, table, key in [('atom', 'atoms', 'atom_id'), ('artifact', 'artifacts', 'artifact_id')]:
                if db is None:
                    record[prefix+'_latest'] = None
                    record[prefix+'_exact_eligible'] = None
                    continue
                rows = db.execute(f'''SELECT a.{key}::text AS identity,a.status,a.is_publishable,
                    v.version_id::text,v.content_hash,v.trust_tier,
                    {prefix}_is_publishable(a.{key}) AS native_eligible
                    FROM {table} a JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s AND v.is_latest''', (spec.fqdn,)).fetchall()
                record[prefix+'_latest'] = rows
                record[prefix+'_exact_eligible'] = len(rows)==1 and rows[0]['content_hash']==spec.content_hash and rows[0]['status']=='approved' and rows[0]['is_publishable'] and rows[0]['native_eligible'] and rows[0]['trust_tier'] in (1,2,3)
            records.append(record)
    return dict(read_only=True, catalog_mutations=0, approved=False, graph_digest=digest,
                catalog_checked=not offline, graph_nodes=len(graph.nodes), direct_providers=len(direct), generic_providers=len(GENERIC),
                inventoried_providers=len(records), complete_transitive_closure=False,
                both_catalogs_exact_eligible=None if offline else sum(r['atom_exact_eligible'] and r['artifact_exact_eligible'] for r in records),
                full_graph_evidence_sha256=sha(evidence_path), providers=records,
                remaining_gates=[
                    'Review complete spatial recipe contracts across training, validation and inference; explicit consumers are connected.',
                    'Validation and checkpoint operations are explicit; review lifecycle contracts and per-epoch invocation evidence.',
                    'Review complete transitive dependencies including local imports, model methods and callbacks.',
                    'Complete exact catalog interfaces, parameters, descriptions, references and Tier 3 audit evidence.',
                    'Qualify corruption rejection, rollback and idempotent provider publication.',
                    'Stage corrected version under original identity, execute stored graph, publish and verify production selection.'],
                limitations=['Provider inventory does not certify semantic approval or complete dependency closure.',
                             'Only two three-epoch training windows have full graph source evidence.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify local sources only; catalog state stays unknown")
    report = inventory(offline=parser.parse_args().offline)
    (ROOT/'docs/reviews/dsb_decomposed_provider_inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('providers','remaining_gates','limitations')}, indent=2))

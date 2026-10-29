"""Read-only reuse and evidence audit for the original DSB competition CDG.

Reads catalog algorithm records and implementation files only, never datasets.
This inventory is not numerical qualification or promotion approval.
"""
import hashlib
import json
from pathlib import Path

from sciona.dsb_execution import build_dsb_raw_training_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.audit_dsb_execution_coverage import STAGES

FQDN = 'cdg.competition.solution.kaggle.dsb2017_1st'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    graph = build_dsb_raw_training_graph()
    digest, _, _ = encode_execution_graph(graph)
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        catalog = db.execute('''
            SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
                   v.version_id::text,v.content_hash,v.trust_tier,
                   (SELECT count(*) FROM artifact_cdg_nodes n WHERE n.version_id=v.version_id) AS nodes,
                   EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn IN (%s,%s) AND v.is_latest ORDER BY a.fqdn
        ''', (FQDN, FQDN + '.execution')).fetchall()
        if len(catalog) != 2:
            raise ValueError('Expected original and derivative latest versions')
        derivative = next(r for r in catalog if r['fqdn'].endswith('.execution'))
        if derivative['content_hash'] != digest or derivative['nodes'] != len(graph.nodes):
            raise ValueError('Local builder differs from current derivative')
        original = next(r for r in catalog if r['fqdn'] == FQDN)
        original_nodes = db.execute('SELECT node_id FROM artifact_cdg_nodes WHERE version_id=%s', (original['version_id'],)).fetchall()
        if {r['node_id'] for r in original_nodes} != set(STAGES):
            raise ValueError('Original source-stage inventory changed')
        bindings = db.execute('''
            SELECT b.node_id,b.bound_artifact_fqdn,b.bound_version_content_hash,b.status,
                   a.atom_id::text,a.status AS legacy_status,a.is_publishable AS legacy_publishable,
                   atom_is_publishable(a.atom_id) AS legacy_native_eligible,
                   artifact_is_publishable(a.atom_id) AS canonical_native_eligible
            FROM artifact_cdg_bindings b LEFT JOIN atoms a ON a.fqdn=b.bound_artifact_fqdn
            WHERE b.version_id=%s ORDER BY b.node_id
        ''', (derivative['version_id'],)).fetchall()
        if len(bindings) != len(graph.nodes):
            raise ValueError('Derivative binding coverage differs')
        metadata = []
        for binding in bindings:
            entry = dict(node_id=binding['node_id'])
            for prefix, key in [('atom','atom_id'),('artifact','artifact_id')]:
                counts = {}
                for suffix in ('parameters','descriptions','references','audit_rollups'):
                    counts[suffix] = db.execute(f'SELECT count(*) AS n FROM {prefix}_{suffix} WHERE {key}=%s',
                                                (binding['atom_id'],)).fetchone()['n']
                entry[prefix] = counts
            metadata.append(entry)
    retained = json.loads((ROOT / 'docs/reviews/competition_dsb_full_source_execution.json').read_text())
    freshness = []
    for key, base in [('implementation_sha256', ROOT), ('provider_sha256', ROOT.parent)]:
        for name, expected in retained[key].items():
            path = base / name
            freshness.append(dict(source=name, matches=path.exists() and sha(path) == expected))
    if retained['execution_graph_sha256'] != digest:
        raise ValueError('Retained execution is for a different graph')
    groups = [
        dict(nodes=['preprocess', 'raw_preparation'], classification='recipe_adapter_with_reusable_geometry',
             finding='Lung segmentation, masking, spatial resampling, cropping, coordinate conversion and population routing are bundled.',
             expose=['Separate spatial resampling and coordinate transforms from lung-specific segmentation and mask thresholds.',
                     'Keep world versus voxel coordinate adapters explicit and route training/validation populations without hidden file lookup.'],
             invariants=['Retain effective spacing, crop bounds and annotation-frame transforms.',
                         'Preserve train/inference mask differences and caller-provided proposal coordinates.']),
        dict(nodes=['prediction'], classification='recipe_adapter_with_reusable_inference_operations',
             finding='One node tiles, runs the detector, combines tiles, decodes boxes, filters, suppresses, crops and scores.',
             expose=['Explicit tile/grid outputs and reconstruction inputs; batched model inference with explicit model state.',
                     'Separate score filtering, cubic-box suppression, crop selection, classifier scoring and noisy-OR reduction.'],
             invariants=['Preserve tile order, margin removal and padded-grid extent.',
                         'Keep strict confidence filter before NMS, top-k order, empty-proposal behavior and both score outputs.',
                         'Network architecture and coordinate channels remain model-specific contracts.']),
        dict(nodes=['training_batches', 'validation_batches'], classification='recipe_adapter_with_reusable_sampling',
             finding='Factories hide size multiplicities, sequential softmax sampling, crops, augmentation and anchor assignment.',
             expose=['Separate generic threshold-based row repetition and sequential weighted sampling from anatomical units and schedules.',
                     'Expose crop/augmentation/anchor operations with explicit RNG and retain distinct training/validation policies.'],
             invariants=['Strict thresholds and additive repetition counts, including omitted rows below all thresholds.',
                         'Sequential probability floor and RNG consumption remain source-compatible.',
                         'Retain source augmentation conventions, including documented coordinate asymmetries.']),
        dict(nodes=['training'], classification='stateful_training_recipe',
             finding='One node restores models, owns optimizers, schedules epochs, computes losses, validates and exports a checkpoint.',
             expose=['Separate runtime restoration, source pass scheduling, differentiable loss operations, validation and checkpoint export.',
                     'Keep the stateful loop explicit; do not replace consecutive training with repeated warm restarts.'],
             invariants=['Preserve shared model parameters, optimizer momentum, RNG state and batch-normalization modes across epochs.',
                         'Preserve detector hard-negative mining, classifier miss penalty and learned noisy-OR baseline.',
                         'Retain zero-learning-rate initialization, schedule boundaries and all validation/metric outputs.']),
        dict(nodes=['models', 'initialize_classifier'], classification='model_specific_state_adapters',
             finding='Restoration and detector-to-classifier transfer require exact architectures and checkpoint schemas.',
             expose=['Declare architecture and state compatibility explicitly; reuse generic state handling only where contracts match.'],
             invariants=['Separate inference detector state from the trained classifier state.',
                         'Preserve shared-weight transfer, head initialization and model-only checkpoint semantics.']),
    ]
    covered = [n for group in groups for n in group['nodes']]
    if len(covered) != len(set(covered)) or set(covered) != {n.node_id for n in graph.nodes}:
        raise ValueError('Reuse inventory must cover every derivative node exactly once')
    return dict(read_only=True, catalog_mutations=0, approved=False, catalog=catalog,
                original_stage_implementations=STAGES,
                derivative_graph_sha256=digest, direct_bindings=bindings, provider_metadata_counts=metadata,
                direct_graph_providers={n.node_id:n.matched_primitive for n in graph.nodes},
                retained_evidence_sha256=sha(ROOT/'docs/reviews/competition_dsb_full_source_execution.json'),
                evidence_freshness=freshness, stale_sources=[r['source'] for r in freshness if not r['matches']],
                retained_synthetic_cases=retained['cases'], decomposition_required=groups,
                qualification_limits=['Existing full-graph evidence covers two three-epoch windows, not complete competition training.',
                    'World-coordinate preparation is covered end-to-end; voxel preparation has separate component evidence.',
                    'Explicit runtime checkpoints, proposal arrays and sample orders remain caller requirements.',
                    'No predictive-quality, anatomical transfer, historical-library or causal-validity claim.'],
                promotion_gates=['Qualify and publish separately reusable numerical providers with exact interfaces and dependencies.',
                    'Preserve numerical behavior, optimizer continuity, model-state sharing and random streams during decomposition.',
                    'Independently compare the complete corrected stored graph against pinned source.',
                    'Preserve original identity/version history; prove corruption rejection and transactional rollback.',
                    'Verify native catalog eligibility and production-selected stored execution.'],
                tier_policy='Automated Tier 3 only; Tier 2 needs community usage and Tier 1 needs human review.',
                auditor_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/dsb_reuse_scope.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(catalog=report['catalog'], stale_sources=report['stale_sources'],
                         audited_nodes=len(report['direct_bindings']))))

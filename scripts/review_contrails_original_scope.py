"""Reconstruct original Contrails scope and map all five stages to source code."""
import hashlib
import json
from pathlib import Path
from sciona.competition_import import _canonical, _id
from scripts.audit_contrails_original_readiness import ROOT, connect, sha, SOURCE_VERSION
from scripts.review_text_pair_original_scope import anchor
from scripts.review_contrails_reuse_boundaries import review as reuse_review

SOURCE_HASH = '6f8962c26d57adbdead9c31a1ed95d7bc31d66bf4e042f5da564d0c8c6d23dee'
STAGES = {
    'temporal_windowing': dict(
        realization='Accept four selected ordered time frames per example; the single branch uses frame3 and the temporal branch uses all four.',
        correction='Replace the inaccurate five-frame window claim. Upstream selection, time spacing, input decoding and alignment are caller responsibilities.',
        anchors=[('contrails_runtime','prepare'), ('contrails_prediction','prepare_inference')]),
    'false_color_generation': dict(
        realization='Apply the exact source inverted three-channel thermal-difference transform with fixed offsets/scales and no clipping, then source branch-specific resizing.',
        correction='This is a physical encoding assumption, not generic RGB normalization. Callers must preserve the channel scale and order; shape validation alone does not establish them.',
        anchors=[('contrails_preprocessing','ash_color'), ('contrails_preparation','prepare_training_example'), ('contrails_prediction','prepare_inference')]),
    '2_5d_stacking': dict(
        realization='Arrange four three-channel frames into a spatial panel with frame order3,0 / 1,2. Keep temporal quadrants fixed under spatial augmentation.',
        correction='Replace the inaccurate fifteen-channel temporal stacking claim. The temporal network receives a three-channel spatial panel; the other branch receives a single three-channel frame.',
        anchors=[('contrails_temporal','create_four_panels'), ('contrails_temporal','create_four_panels_tta')]),
    'u_net_architecture': dict(
        realization='Train both full MaxViT/U-Net branches across every selected fold, preserving source decoder/head differences, shifted-label geometry, augmented-loss masking, AdamW, scheduling, validation-break timing and terminal checkpoint reload.',
        correction='The source realization selects MaxViT rather than an interchangeable ConvNeXt alternative. It does not operate on a fifteen-channel input. Explicit shortened epochs and random initialization qualify execution only, not historical accuracy.',
        anchors=[('contrails_single_model','Model'), ('contrails_temporal_model','Model'), ('contrails_lifecycle','run_lifecycle'), ('contrails_fold','train_fold'), ('contrails_epoch','train_epoch')]),
    'temporal_tta': dict(
        realization='Run eight-way spatial D4 probability averaging for each selected checkpoint; combine folds and branches using the exact unnormalized source weights and encode the strict0.5 decision as column-major run lengths.',
        correction='Replace temporal-window shifting with spatial rotations/flips. Threshold diagnostics do not replace the fixed final threshold, and weighted outputs are not calibrated probabilities.',
        anchors=[('contrails_preprocessing','SpatialTTA'), ('contrails_temporal_inference','Model'), ('contrails_single_inference','Model'), ('contrails_prediction','ensemble_probabilities'), ('contrails_prediction','encode_mask')]),
}


def review():
    historical_path = ROOT/'docs/reviews/competition_contrails_semantic_review.json'
    historical = json.loads(historical_path.read_text())
    if historical['source_version_id'] != SOURCE_VERSION or historical['source_hash'] != SOURCE_HASH:
        raise ValueError('Historical source review identity changed')
    for name, digest in historical['evidence_sha256'].items():
        if sha(ROOT/'docs/reviews'/name) != digest:
            raise ValueError('Historical source comparison evidence changed')
    for name, digest in historical['auxiliary_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Historical license or default evidence changed')
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        rows = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'", (SOURCE_VERSION,)).fetchall()
        if len(rows) != 1:
            raise ValueError('Unique original snapshot required')
        snapshot = rows[0]['details']['snapshot']
        digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
        fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
        if digest != SOURCE_HASH or _id(fqdn+'/'+digest) != SOURCE_VERSION:
            raise ValueError('Original source identity differs')
    stages = snapshot['template']['stages']
    if len(stages) != 5 or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Exact original five-stage coverage required')
    reuse = reuse_review()
    return dict(format='contrails-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, review_source='automated',
        scope='Source-scope review only; full-model execution and publication remain separate requirements.',
        artifact_id=_id(fqdn), fqdn=fqdn, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
        source_snapshot_reconstructed=True, source_commit=historical['source_commit'],
        source_stage_review=[dict(stage_id=s['stage_id'], realization=STAGES[s['stage_id']]['realization'],
            correction=STAGES[s['stage_id']]['correction'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision=reuse['reuse_decision'], limitations=[r['limitation'] for r in reuse['boundaries']],
        historical_source_review_sha256=sha(historical_path),
        reuse_reviewer_sha256=reuse['reviewer_sha256'], reviewer_sha256=sha(Path(__file__)),
        remaining=['Complete both full-model variants with explicit bounded resources and dependency integrity.',
                   'Review provider contracts and reconcile metadata with rollback gates.',
                   'Stage and execute the corrected stored original; qualify publication and served selection.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/contrails_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(report['source_stage_review']), approved=False)))

"""Reconstruct original DFDC scope and map all five stages to source code."""
import hashlib
import json
from pathlib import Path
from sciona.competition_import import _canonical, _id
from scripts.audit_dfdc_original_readiness import ROOT, connect, sha, SOURCE_VERSION
from scripts.review_text_pair_original_scope import anchor
from scripts.review_dfdc_reuse_boundaries import review as reuse_review

SOURCE_HASH = '66f965f75d83ba4ec2daf15afcfa7a657589938dc98c00722b9ccea7eec92679'
STAGES = {
    'frame_extraction': dict(
        realization='Accept decoded uint8 RGB clips with explicit order, then retain source frame selection and empty-input fallback semantics.',
        correction='Media decoding is a caller responsibility. This implementation does not infer frame timing or authenticate historical media membership.',
        anchors=[('dfdc_runtime','_frames'),('dfdc_frame_sampling','select_frames')]),
    'face_detection': dict(
        realization='Use the actual three-network MTCNN cascade with explicit detector state policy and stage-specific training/inference preparation.',
        correction='Select the source MTCNN realization rather than RetinaFace. Constructed synthetic states qualify execution, not pretrained face detection quality.',
        anchors=[('dfdc_detector','build_detector'),('dfdc_training_crops','build_preprocessing_detector'),('dfdc_faces','extract_faces')]),
    'face_alignment_cropping': dict(
        realization='Apply source margin crops without inference landmark alignment; use five-point preprocessing and a separate native sixty-eight-point predictor for training occlusion.',
        correction='Do not claim inference alignment. Landmark count does not prove semantic order. Ordered polygon occlusion is not a generic geometric convex-hull construction.',
        anchors=[('dfdc_faces','expanded_crop'),('dfdc_crop_annotations','annotate_pair'),('dfdc_hull_backend','load_hull_backend')]),
    'deep_fake_classification': dict(
        realization='Execute full EfficientNet-B7 paired-record training, source population balancing, smoothed balanced loss, SGD, scheduling and numbered checkpoint collection with an explicit run/request plan.',
        correction='Select B7 rather than an interchangeable Se-ResNeXt. Source forty-epoch runs cannot supply zero-based checkpoint forty; require an explicit executable adaptation without extension or relabeling. Synthetic shortened training does not establish historical accuracy.',
        anchors=[('dfdc_classifier','build_classifier'),('dfdc_ensemble_training','train_ensemble'),('dfdc_training','train'),('dfdc_selection','selection_plan')]),
    'temporal_aggregation': dict(
        realization='Apply confidence heuristics to face predictions separately per selected checkpoint, then average checkpoint scores, preserving source count thresholds and explicit fallback status.',
        correction='This is not generic temporal averaging. A fallback score0.5 can denote missing faces or execution failure; consumers must inspect status. Scores are not calibrated for other domains.',
        anchors=[('dfdc_inference_primitives','confident_strategy'),('dfdc_prediction','predict_video'),('dfdc_ensemble_prediction','predict_selected')]),
}


def review():
    historical_path = ROOT/'docs/reviews/competition_dfdc_semantic_review.json'
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
    return dict(format='dfdc-original-scope.v1', passed=True, approved=False,
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
        remaining=['Complete the full five-run/seven-checkpoint graph with bounded resources and dependency integrity.',
                   'Review provider contracts and reconcile metadata with rollback gates.',
                   'Stage and execute the corrected stored original; qualify publication and served selection.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/dfdc_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(report['source_stage_review']), approved=False)))

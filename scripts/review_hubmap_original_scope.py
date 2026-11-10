"""Reconstruct original HuBMAP scope and map all six stages to source code."""
import hashlib
import json
from pathlib import Path
from sciona.competition_import import _canonical, _id
from scripts.audit_hubmap_original_readiness import ROOT, connect, sha, SOURCE_VERSION
from scripts.review_text_pair_original_scope import anchor
from scripts.review_hubmap_reuse_boundaries import review as reuse_review

SOURCE_HASH = '6f290e9ea9794b85f83ff3e52034a41a1c400f84d2f3f5182beb65d837664468'
STAGES = {
    'tiling': dict(
        realization='Prepare decoded RGB/binary raster pairs using source unshifted and half-shifted grids, mask-area ordering, JPEG roundtrip and selected tile populations.',
        correction='Raster decoding is a caller boundary. Grid padding, source ordering and lossy preprocessing remain explicit; no arbitrary interchangeable tiler is claimed.',
        anchors=[('hubmap_tiling','RawTiles'),('hubmap_raw_population','prepare_population'),('hubmap_preparation','prepare_tile')]),
    'stain_normalization': dict(
        realization='Use source RGB/BGR conversion, JPEG preprocessing and fixed channel normalization in augmentation and inference.',
        correction='The intake stain-normalization claim is unsupported by the qualified source. Do not represent fixed channel normalization as stain normalization. This correction must be explicit in the corrected graph.',
        anchors=[('hubmap_preparation','prepare_tile'),('hubmap_augmentation','augment'),('hubmap_inference_tiles','InferenceTiles')]),
    'segmentation_architecture': dict(
        realization='Use full SE-ResNeXt101 encoder with custom decoder, attention and hypercolumns; execute initial folds, model restoration, generated pseudo masks and fresh retraining folds.',
        correction='Replace the intake DeepLabV3+/ResNeSt50 description with the source architecture. Initial training and pseudo-label retraining are required lifecycle stages omitted by the intake. Explicit encoder states are required; no automatic pretrained download.',
        anchors=[('hubmap_network','UNET_SERESNEXT101'),('hubmap_encoder','build_encoder'),('hubmap_lifecycle','train_fold'),('hubmap_graph_runtime','retraining')]),
    'lovasz_bce_loss': dict(
        realization='Train with binary BCE and per-image Lovasz hinge plus four nonempty-only deep supervision terms weighted0.1 and image-classification BCE.',
        correction='This is binary Lovasz hinge, not multiclass Lovasz-Softmax. Preserve auxiliary losses and their empty-mask gradient behavior.',
        anchors=[('hubmap_losses','binary_lovasz_hinge'),('hubmap_losses','training_loss'),('hubmap_training','train_batch')]),
    'tta_on_tiles': dict(
        realization='Use four flip views for pseudo labels and three for final prediction; average models and views before resize and strict binary thresholding.',
        correction='Pseudo and final inference policies differ. Batch-wide classification shortcut and ensemble order remain source assumptions; binary masks are not calibrated probabilities.',
        anchors=[('hubmap_graph_runtime','pseudo_labels'),('hubmap_graph_runtime','final_prediction'),('hubmap_inference','predict_batch'),('hubmap_network','UNET_SERESNEXT101.forward')]),
    'gigapixel_stitching': dict(
        realization='Prepare padded inference grids in bounded batches and write predicted interiors into a full-size binary output mask.',
        correction='Full decoded source and output arrays remain resident. No TIFF streaming, unbounded-size memory guarantee or source-codec qualification is implied.',
        anchors=[('hubmap_raw_inference','predict_slide'),('hubmap_inference_tiles','InferenceTiles')]),
}


def review():
    import subprocess
    from scripts.promote_hubmap_execution import review as historical_review
    historical_path = ROOT/'docs/reviews/competition_hubmap_source_review.json'
    historical = json.loads(historical_path.read_text())
    if historical['source_version_id'] != SOURCE_VERSION or historical['source_content_hash'] != SOURCE_HASH:
        raise ValueError('Historical source identity changed')
    evidence = historical_review(ROOT)[-1]
    for name, digest in evidence.items():
        path = 'docs/reviews/'+name
        raw = subprocess.check_output(['git','show','993e042d:'+path],cwd=ROOT)
        if raw != (ROOT/path).read_bytes() or sha(ROOT/path) != digest:
            raise ValueError('Immutable historical evidence changed')
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
    if len(stages) != 6 or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Exact original six-stage coverage required')
    reuse = reuse_review()
    return dict(format='hubmap-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, review_source='automated',
        scope='Source-scope review only; full-model execution and publication remain separate requirements.',
        artifact_id=_id(fqdn), fqdn=fqdn, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
        source_snapshot_reconstructed=True, source_commit=historical['source_commit'],
        source_stage_review=[dict(stage_id=s['stage_id'], realization=STAGES[s['stage_id']]['realization'],
            correction=STAGES[s['stage_id']]['correction'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision=reuse['reuse_decision'], limitations=[r['limitation'] for r in reuse['boundaries']],
        historical_source_review_sha256=sha(historical_path),historical_evidence_sha256=evidence,
        reuse_reviewer_sha256=reuse['reviewer_sha256'], reviewer_sha256=sha(Path(__file__)),
        remaining=['Execute full initial and retraining networks, pseudo handoff and final prediction with explicit fold coverage, source-sized geometry and bounded resources.',
                   'Review provider contracts and reconcile metadata with rollback gates.',
                   'Stage and execute the corrected stored original; qualify publication and served selection.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/hubmap_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(report['source_stage_review']), approved=False)))

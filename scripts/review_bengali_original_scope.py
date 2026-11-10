"""Map every original Bengali intake stage to the reconstructed winner recipe.

Source coverage and execution do not resolve outstanding license permissions.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from sciona.competition_import import _canonical, _id
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.review_text_pair_original_scope import anchor
from scripts import bengali_complete_evidence as execution

COMMIT = '55af25e366b13c79185a897a763ef86cab510e4f'
SOURCE_VERSION = '0c0997e0-c5f7-509c-9068-dcf00a167f56'
SOURCE_HASH = 'd9dfe89a2b146a6ee459b864b7d93e998a2e368b6ac5154b8aa088f11cfd4c82'
STAGES = {
    'image_preprocessing': dict(
        realization='Keep B7 grayscale geometry137x236, replicate RGB and use explicit normalization. The separate font/GAN path crops six columns per side, resizes224 square and normalizes with mean/std0.5.',
        correction='Remove mandatory inversion and content-bounding-box crop. Winner evidence rejects cropping for the B7 path; mandatory inversion is unsupported. The fixed font-path crop is a different operation.',
        anchors=[('grapheme_b7_preprocessing', 'prepare_classifier_images'), ('bengali_preprocessing', 'prepare_handwriting')]),
    'heavy_augmentation': dict(
        realization='Use reconstructed SVHN AutoAugment for B7 and source-specific font augmentation with explicit random streams. Preserve separate handwriting and font preprocessing.',
        correction='CutMix, MixUp and GridMask are unsupported intake claims. Do not attribute mixed-label training to this reconstruction.',
        anchors=[('svhn_autoaugment', 'SVHNAutoAugment'), ('grapheme_workflow_runtime', 'train_b7'), ('grapheme_workflow_runtime', 'train_font')]),
    'multihead_backbone': dict(
        realization='Train separate EfficientNet-B7 OOD1295 and seen14784 classifiers; train two B0 font classifiers and two classifier-guided CycleGAN branches.',
        correction='Replace the SE-ResNeXt50 three-head description. The font classifier uses the implemented B0 width1280, not the paper width2560. All six fits belong to the complete recipe.',
        anchors=[('grapheme_b7_head', 'GraphemeB7Head'), ('grapheme_workflow_runtime', 'train_font'), ('grapheme_workflow_runtime', 'train_generator')]),
    'weighted_multitask_loss': dict(
        realization='Use hard-negative BCE for OOD, cross entropy for seen/font classifiers, and bidirectional adversarial, cycle and frozen-classifier guidance losses for domain translation.',
        correction='The2:1:1 component weights belong to the evaluation/calibration metric, not a three-head classification training loss. Keep metric and training objectives distinct.',
        anchors=[('bengali_ood_loss', 'ood_training_loss'), ('grapheme_workflow_runtime', 'train_b7'), ('bengali_gan_training', 'CycleGANTraining'), ('grapheme_routing_calibration', 'calibrate_grapheme_router')]),
    'ohem_training': dict(
        realization='For each OOD example include its positive class and negative classes whose detached sigmoid exceeds0.1; normalize by selected class count then average examples.',
        correction='This is class-level hard-negative selection, not top-k sample OHEM. Every positive target contributes; no claim that only hardest batch examples receive updates.',
        anchors=[('bengali_ood_loss', 'ood_training_loss')]),
    'soft_voting_ensemble': dict(
        realization='Softmax and marginalize each font branch, combine component scores, fit a confidence threshold on separate weighted calibration rows, route low-confidence unknown alternate classes, and decode canonical components.',
        correction='Replace simple seed-averaging across three heads with OOD/seen/two-font routing and known-class fallback. Threshold grid, weights and seen softmax decoder are explicit reconstruction choices; exact historical values remain unverified.',
        anchors=[('bengali_unseen_ensemble', 'marginal_predictions'), ('grapheme_routing_calibration', 'calibrate_grapheme_router'), ('bengali_routing', 'route_predictions'), ('grapheme_routing_calibration', 'submission_components')]),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed(path):
    raw = subprocess.check_output(['git', 'show', COMMIT+':'+path], cwd=ROOT)
    if raw != (ROOT/path).read_bytes():
        raise ValueError('Retained source review changed: '+path)
    return json.loads(raw)


def review():
    triage_path = 'docs/reviews/competition_bengali_grapheme_source_triage.json'
    reuse_path = 'docs/reviews/bengali_reuse_boundaries.json'
    providers_path = 'docs/reviews/bengali_provider_contracts.json'
    triage, reuse, providers = map(committed, (triage_path, reuse_path, providers_path))
    if triage['intake_version_id'] != SOURCE_VERSION or triage['intake_content_hash'] != SOURCE_HASH:
        raise ValueError('Retained original identity differs')
    for path, digest in reuse['implementation_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Reviewed reuse implementation changed')
    for path, digest in providers['implementation_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Provider contract implementation changed')
    qualification = execution.review('6fb098a30429ccde1c4beb39daed250a90d1157a')
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        rows = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s "
            "AND runner_version='competition-intake.v1'", (SOURCE_VERSION,)).fetchall()
        if len(rows) != 1:
            raise ValueError('Unique original source snapshot required')
        snapshot = rows[0]['details']['snapshot']
        digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
        fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
        if digest != SOURCE_HASH or _id(fqdn+'/'+digest) != SOURCE_VERSION:
            raise ValueError('Original source snapshot differs')
    stages = snapshot['template']['stages']
    if len(stages) != 6 or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Every original stage must be reviewed')
    return dict(format='bengali-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, review_source='automated', source_snapshot_reconstructed=True,
        scope='Complete intake-stage correction mapping with retained source evidence; not source-license clearance or publication.',
        artifact_id=_id(fqdn), fqdn=fqdn, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
        source_review_commit=COMMIT,
        source_stage_review=[dict(stage_id=s['stage_id'], realization=STAGES[s['stage_id']]['realization'],
            correction=STAGES[s['stage_id']]['correction'],
            implementation_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        omitted_original_lifecycle=['Separate OOD and seen B7 training', 'Two font-classifier fits',
            'Two classifier-guided CycleGAN fits', 'Fresh checkpoint inference for calibration and query',
            'Weighted calibration and known-class fallback'],
        full_execution=qualification, reuse_decision=reuse['reuse_decision'],
        evidence_sha256={p: sha(ROOT/p) for p in (triage_path, reuse_path, providers_path)},
        limitations=[b['limitation'] for b in reuse['boundaries']]+qualification['limits'],
        outstanding_permissions=[
            'Training notebook version4 has an indexed Apache2 declaration; classifier and submission notebook licenses remain unresolved.',
            'Review upstream attribution/notices and candidate B7 initialization permissions separately. A compatible checkpoint is not historical provenance.',
        ],
        remaining=['Resolve permissions and explicitly retain unrecovered historical policy limits.',
            'Complete provider semantic/dependency metadata and qualify approval transaction gates.',
            'Stage and execute the complete corrected stored original; prove publication rollback, repeat and served selection.'],
        reviewer_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/bengali_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, stages=len(result['source_stage_review']))))

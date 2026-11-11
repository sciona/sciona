"""Map every original Cornell stage to the source-qualified current recipe."""
import hashlib
import json
import subprocess
from pathlib import Path

from sciona.competition_import import _canonical, _id
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.audit_cornell_original_readiness import sha, SOURCE_VERSION
from scripts.review_cornell_provider_contracts import review as provider_review
from scripts.review_text_pair_original_scope import anchor
from scripts.verify_cornell_metadata import verify

SOURCE_HASH = '0e75554c472b820a39e97f3807ebc13ebd7d915af068b36ca4d364997b9b1690'
STAGES = {
    'audio_conversion': dict(
        realization='Accept caller-prepared mono 32000 Hz arrays; execute the source waveform STFT and log-mel representation inside each full sound-event model.',
        limits='No file decoding, resampling, five-minute input requirement or saved spectrogram images. Representation is part of the model and follows waveform augmentation during training, correcting the intake linear ordering.',
        anchors=[('cornell_contract', 'prepare'), ('cornell_model', 'SourceModel')]),
    'background_noise_augmentation': dict(
        realization='Run the pinned three-transform or eight-transform historical audio augmentation recipe with supplied background and short-noise arrays before source training crops; retain class-balanced replacement sampling and training drop-last.',
        limits='No automatic external noise acquisition or inferred noise semantics. Explicit per-instance random state replaces worker entropy. Reject silent/nonfinite or short banks; historical Gaussian-SNR behavior is retained without reinterpreting it as modern decibel semantics.',
        anchors=[('cornell_augmentation', 'Augmentation'), ('cornell_population', 'batches'), ('cornell_sampling', 'sample_waveform')]),
    'sound_event_detection_sed_model': dict(
        realization='Train thirteen full DenseNet121 attention sound-event models in seventeen phases, including four model-only continuations with fresh optimizer state; retain source losses, schedule, two-clip validation, checkpoint selection and reload.',
        limits='Correct the ResNeSt-50 claim. Fixed 264 output slots and explicit fourfold/fivefold populations remain recipe policy. Caller chooses epochs and initialization; current synthetic qualification uses two epochs per phase. Best-score/latest-retained-tie selection replaces hand-picked historical checkpoints. No pretrained or predictive-accuracy parity.',
        anchors=[('cornell_model', 'SourceModel'), ('cornell_lifecycle', 'execute'), ('cornell_training', 'train_phase')]),
    'thresholding_optimization': dict(
        realization='Apply fixed 0.3 clip and frame thresholds, require four of thirteen model votes for an interval, and four already-selected intervals for a whole-record decision.',
        limits='Correct the learned per-species threshold claim: no threshold fitting, probability calibration or label taxonomy is inferred. Independent array voting supports singleton/empty events and excludes padding/endpoints with half-open intervals.',
        anchors=[('cornell_voting', 'vote')]),
    'inference_chunking': dict(
        realization='Predict nonoverlapping thirty-second chunks using all thirteen selected models, average ten evaluation copies per model, then aggregate frame events into five-second intervals and whole-record decisions.',
        limits='Correct overlapping five-second model-window inference. Evaluation disables spectrogram masking despite ten copies. Partial final intervals are supported. Outputs are ordered generic class-slot decisions, without named submission formatting or persisted trained models.',
        anchors=[('cornell_lifecycle', 'predict'), ('cornell_voting', 'vote'), ('cornell_contract', 'execute')]),
}


def review():
    contracts = provider_review()
    verify()
    historical_path = 'docs/reviews/competition_cornell_semantic_review.json'
    raw = subprocess.check_output(['git', 'show', '6a4be8eb5029b80913515743bada228579e5ce7d:'+historical_path], cwd=ROOT)
    if raw != (ROOT/historical_path).read_bytes():
        raise ValueError('Historical source review changed')
    historical = json.loads(raw)
    for name, digest in historical['evidence_sha256'].items():
        if sha(ROOT/'docs/reviews'/name) != digest:
            raise ValueError('Historical source comparison evidence changed')
    for name, digest in historical['auxiliary_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Historical reviewer or license evidence changed')
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        rows = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'", (SOURCE_VERSION,)).fetchall()
        if len(rows) != 1:
            raise ValueError('Unique original intake snapshot required')
        snapshot = rows[0]['details']['snapshot']
        digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
        fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
        if digest != SOURCE_HASH or _id(fqdn+'/'+digest) != SOURCE_VERSION:
            raise ValueError('Original source identity differs')
        identity = db.execute('SELECT a.artifact_id::text,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s', (SOURCE_VERSION,)).fetchone()
        if identity != dict(artifact_id=_id(fqdn), fqdn=fqdn, content_hash=SOURCE_HASH):
            raise ValueError('Original catalog identity differs')
    stages = snapshot['template']['stages']
    if len(stages) != 5 or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Exact five-stage coverage required')
    if {s['intake'] for s in historical['intake_stage_mapping']} != set(STAGES):
        raise ValueError('Historical source coverage differs')
    return dict(format='cornell-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, review_source='automated', proposed_tier=3,
        artifact_id=_id(fqdn), fqdn=fqdn, source_version_id=SOURCE_VERSION,
        source_content_hash=SOURCE_HASH, source_snapshot_reconstructed=True,
        source_commit=historical['source_commits']['winning_solver'],
        source_stage_review=[dict(stage_id=s['stage_id'], realization=STAGES[s['stage_id']]['realization'],
            limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision=contracts['reuse_decision'], limitations=contracts['limitations'],
        historical_evidence_scope='Immutable source-comparison record; historical package versions are not claimed to match the current runtime. Current contracts and full graph execution separately requalify that runtime.',
        source_review_sha256=sha(ROOT/historical_path),
        provider_contracts_sha256=sha(ROOT/'docs/reviews/cornell_provider_contracts.json'),
        current_execution_sha256=sha(ROOT/'docs/reviews/cornell_current_graph.json'),
        reviewer_sha256=sha(Path(__file__)),
        remaining=['Corrected original stored graph execution and immutable evidence.',
                   'Transactional staging/publication gates and served production selection.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/cornell_original_scope.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(report['source_stage_review']), approved=False)))

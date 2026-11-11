"""Map the exact original intake to the qualified subject-separated waveform/PSD realization."""
import hashlib
import json
import subprocess

from sciona.competition_import import FORMAT, _canonical, _id
from scripts.review_text_pair_original_scope import anchor
from scripts.review_biosignal_provider_semantics import review as provider_review
from scripts.validate_biosignal_complete_runtime import ROOT, sha

ORIGINAL = '1587d081-d47c-5595-8bed-d3d3aa6465af'
ORIGINAL_HASH = '2ab7a80c92296d3fffeb6187655e52443caf311086b4cdc3ed1efde5fe4406c8'
IDENTITY = '9b4ce7ba-b074-5d44-8be0-32363376c748'
STAGES = {
    'subject_safe_windowing': dict(
        anchors=[('biosignal_sequence_contract','prepare'), ('biosignal_sequence_features','window_signal'), ('biosignal_complete_stages','prepare')],
        realization='Require aligned signal channels and explicit shared sample rate, then enforce disjoint caller-supplied subject keys across training/calibration/query. Form fixed windows strictly within each recording, preserving recording membership and window counts.',
        limits='Caller supplies meaningful subject IDs and common channel meaning/order/units. No automatic montage or metadata discovery. Incomplete trailing windows are omitted; grouping is subject-disjoint across populations, not an automatic cross-validation scheme.'),
    'spectral_signal_features': dict(
        anchors=[('biosignal_sequence_features','psd_spectrogram'), ('biosignal_sequence_training','Scaling.fit'), ('biosignal_sequence_training','Scaling.transform')],
        realization='Select one-sided density PSD spectrograms with periodic Hann, segment-mean detrending and log1p. Fit recording-balanced per-channel scaling on training windows only, preserving spectral frequency positions and raw waveform branch.',
        limits='Shared training-population channel scaling is explicit; no subject-adaptive centering or fitting on calibration/query signals. No CWT, log-mel, bipolar montage or arbitrary handcrafted-feature branch. Constant scales become one. Overlapping windows repeat samples in scaling statistics.'),
    'deep_signal_backbones': dict(
        anchors=[('biosignal_sequence_network','DualSignalHead'), ('biosignal_sequence_training','fit'), ('biosignal_complete_stages','fit')],
        realization='Select a joint 1D waveform CNN and 2D spectrogram CNN. Pool waveform time and spectrogram time while retaining frequency position, concatenate branches, and train one binary recording-supervised window score with CPU float64 Adam.',
        limits='Independent small CNN architecture, not a pretrained Swin/EfficientNet or sequence-attention branch. Each recording label supervises its windows. Fixed epoch training without early stopping. Features are fully materialized; no large-scale memory or clinical accuracy claim.'),
    'augmentation_and_loss': dict(
        anchors=[('biosignal_sequence_network','mask_spectrogram'), ('biosignal_sequence_training','window_weights'), ('biosignal_sequence_training','fit')],
        realization='Select seeded time/frequency masking of normalized spectrograms and class/recording-balanced BCE. Accumulate globally weighted gradients across batches and update once per complete epoch; waveform branch remains unmasked.',
        limits='Equal recording weight within each class, not equal subject weight. Zero is normalized spectral center, not physical signal absence. No mixup, focal, entmax or noisy-to-clean branch. Class weighting does not establish calibrated probabilities or learned label uncertainty.'),
    'window_aggregation_calibration': dict(
        anchors=[('biosignal_sequence_training','SignalModel.predict'), ('biosignal_complete_stages','calibrate'), ('biosignal_complete_stages','predict')],
        realization='Select mean window probability per recording and separate empirical-F1 threshold tuning through existing domain-independent providers. Apply inclusive threshold with highest-threshold tie policy to ordered query recordings and return scores, classes and counts.',
        limits='Decision-threshold calibration only; no probability-calibration guarantee or independent final accuracy claim. Generic binary recording output, not a named benchmark submission or patient diagnosis. No attention pooling, fold ensemble or rank-averaging branch.'),
}

def review():
    semantics_path = 'docs/reviews/biosignal_provider_semantics.json'
    raw = (ROOT/semantics_path).read_bytes()
    semantics = json.loads(raw)
    if semantics != provider_review():
        raise ValueError('Qualified provider semantics changed')
    directory = ROOT.parent/'sciona-atoms/data/solution_cdgs'
    name = 'biosignal_spectrogram_sequence_classification_topology'
    snapshot = dict(format=FORMAT, template=json.loads((directory/(name+'.json')).read_text()),
        bindings=json.loads((directory/(name+'_bindings.json')).read_text()))
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
    if (digest, _id(fqdn+'/'+digest), _id(fqdn)) != (ORIGINAL_HASH, ORIGINAL, IDENTITY):
        raise ValueError('Local source does not reconstruct the exact original intake')
    stages = snapshot['template']['stages']
    if len(stages) != len(STAGES) or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Every original stage must be covered exactly once')
    return dict(format='biosignal-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, catalog_state_checked=False, source_snapshot_reconstructed=True,
        source_version_id=ORIGINAL, source_content_hash=ORIGINAL_HASH, artifact_id=IDENTITY, fqdn=fqdn,
        provider_semantics_sha256=sha(ROOT/semantics_path), reviewer_sha256=sha(__file__),
        source_stage_review=[dict(source_stage={k:s[k] for k in ('stage_id','description','guarantees')},
            realization=STAGES[s['stage_id']]['realization'], limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision='Reuse two existing empirical-F1 providers without changing their catalog identities or contracts. Four recipe adapters expose preparation, fitting, calibration and query prediction. Units, channel meaning, subject separation, windowing, CNN and recording aggregation remain explicit recipe policy.',
        remaining_requirements=['Recheck original snapshot and preserved derivative against live catalog state.',
            'Reconcile and approve exact provider versions with metadata, dependency and rollback checks.',
            'Bind, store and execute the corrected original, then verify Tier 3 publication and served retrieval.'],
        limits=['Source reconstruction proves exact intake content identity, not current database status or approval.',
            'All five source stages mapped to the selected binary waveform/PSD realization. Generic threshold selection is not calibrated probability, clinical utility or independent accuracy.',
            'Automated synthetic Tier 3 candidate evidence only; no human Tier 1 designation or empirical deployment guarantee.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/biosignal_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(result['source_stage_review']), exact_source=True, approved=False)))

"""Offline semantic contracts bound to the complete biosignal runtime evidence."""
import inspect
import json
import subprocess

from scripts.validate_biosignal_complete_runtime import ROOT, sha, source_inventory
from scripts.review_video_numeric_runtime import review as runtime_review
from sciona.atoms.ml import biosignal_complete_execution as adapters
from scripts.validate_biosignal_complete_runtime import PROVIDERS

EVIDENCE_COMMIT = '0dc1db78'
EVIDENCE = 'docs/reviews/biosignal_complete_runtime.json'
CONTRACTS = {
    'prepare': dict(
        plain='Check signal arrays, shared channel layout and disjoint subject populations.',
        bounds='Exact finite version-1 JSON with training, calibration, unlabeled query and explicit controls. Nonempty rectangular channel/time signals with a common channel count; caller guarantees consistent channel meaning, units and sample rate. Each labeled population has both binary classes. Nonempty subject keys must be disjoint across populations. Validated window/stride/FFT/hop, seed, width, epochs, batch size, learning rate and mask limits. Return detached immutable JSON configuration; no automatic montage or identity discovery.',
        dependencies=[]),
    'fit': dict(
        plain='Fit signal scaling and a waveform/spectrogram model using training recordings only.',
        bounds='Revalidated immutable Prepared configuration. Fixed windows stay within recordings; trailing incomplete samples omitted. Periodic Hann segment-mean-detrended one-sided density PSD with log1p. Recording-balanced training-only channel scaling, constant scales replaced by one. CPU float64 dual 1D waveform/2D spectrogram CNN with frequency-preserving pooling. Seeded time/frequency masks on normalized spectra; class/recording-balanced BCE, Adam and one accumulated update per epoch. All features are materialized in memory; batch size bounds neural forward allocation only. No early stopping, pretrained model, subject-equal weighting or empirical accuracy claim. Trusted model/scaling objects, no portable persistence.',
        dependencies=[]),
    'calibrate': dict(
        plain='Choose a recording-level decision threshold from the separate calibration population.',
        bounds='Unmodified Fitted state and revalidated source configuration. Predict mean window probabilities for calibration recordings without fitting model/scaling. Call the existing generic float64 empirical-F1 provider on one score column and int64 binary labels. Inclusive comparison and highest observed threshold among equal F1 values. Return threshold and calibration count while retaining fitted state. This selects a decision threshold, not calibrated probabilities or unbiased accuracy. Caller owns meaningful subject identities; calibration labels must not be used to train the model.',
        dependencies=['fit_empirical_f1_thresholds']),
    'predict': dict(
        plain='Predict query recordings and apply the fitted decision threshold without training again.',
        bounds='Unmodified Calibrated state. Revalidate configuration, predict ordered mean window query scores and apply the existing inclusive column-threshold provider. Return finite JSON scores/classes/threshold, population and window counts, omitted query samples, epochs and initial/final training loss. Query labels are prohibited. No refit; trusted in-process model state. Predictions are generic binary recording results, not a clinical interpretation, named submission format or independent accuracy claim.',
        dependencies=['apply_column_thresholds']),
}

def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    if raw != subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT):
        raise ValueError('Immutable runtime evidence changed')
    evidence = json.loads(raw)
    if evidence['tests'] != dict(tests=67, failures=0, errors=0, skipped=0) or evidence['passed'] is not True:
        raise ValueError('Complete passing runtime qualification required')
    for path, digest in evidence['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Qualified source changed: '+path)
    if source_inventory() != evidence['providers'] or runtime_review() != evidence['runtime']:
        raise ValueError('Registered source or runtime changed')
    records = {r['runtime_fqdn'].rsplit('.', 1)[1]:r for r in evidence['providers']}
    if {name for name, r in records.items() if r['role'] == 'recipe_adapter'} != set(CONTRACTS):
        raise ValueError('Every provider requires semantic review')
    functions = {f.__name__:f for f in PROVIDERS.values()}
    functions.update({name:getattr(adapters, name) for name in ('prepare','fit','calibrate','predict')})
    providers = []
    for name, r in sorted(records.items()):
        if r['role'] != 'recipe_adapter':
            continue
        contract = CONTRACTS[name]
        signature = inspect.signature(functions[name])
        ports = [dict(direction='input', ordinal=i, name=p.name,
            type_desc=inspect.formatannotation(p.annotation), required=True, default_value_repr='',
            constraints=contract['bounds']) for i,p in enumerate(signature.parameters.values())]
        ports.append(dict(direction='output', ordinal=0, name='result',
            type_desc=inspect.formatannotation(signature.return_annotation), required=True,
            default_value_repr='', constraints=contract['bounds']))
        providers.append(dict(r, plain_description=contract['plain'], validity_bounds=contract['bounds'], ports=ports,
            dependencies=[{key:records[d][key] for key in ('runtime_fqdn','catalog_fqdn','version_id','content_hash')}
                for d in contract['dependencies']]))
    return dict(format='biosignal-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        catalog_mutations=0, catalog_state_checked=False, review_source='automated',
        runtime_evidence_commit=EVIDENCE_COMMIT, runtime_evidence_sha256=sha(ROOT/EVIDENCE),
        reviewer_sha256=sha(__file__), providers=providers,
        reused_numeric_providers=[r for r in records.values() if r["role"] == "numeric"],
        limits=['Offline semantic candidate review, not catalog approval or a human Tier 1 designation.',
            'Two existing generic F1 providers are reused unchanged. Units, population separation, spectral representation, model and aggregation remain recipe policy.',
            '67 synthetic tests cover DFT/PSD numerics, class/recording weights, native execution, F1 oracles and no-refit parity; no clinical or empirical deployment guarantee.',
            'Live catalog reconciliation, original source-scope review, transaction checks and served selection remain required.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/biosignal_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), catalog_state_checked=False, approved=False)))

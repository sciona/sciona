"""Offline semantic contracts bound to the complete image_tabular runtime evidence."""
import inspect
import json
import subprocess

from scripts.validate_image_tabular_complete_runtime import ROOT, sha, source_inventory
from scripts.review_video_numeric_runtime import review as runtime_review
from sciona.atoms.ml import image_tabular_complete_execution as adapters
from scripts.validate_image_tabular_complete_runtime import PROVIDERS

EVIDENCE_COMMIT = 'a03ab602'
EVIDENCE = 'docs/reviews/image_tabular_complete_runtime.json'
CONTRACTS = {
    'prepare': dict(
        plain='Validate images, metadata and separate training, calibration and query groups.',
        bounds='Exact finite version-1 JSON. Normalized RGB images in [0,1], aligned rectangular numeric and categorical metadata, binary labels on training/calibration only. Both labeled populations contain both classes. Nonempty caller-supplied groups are disjoint across populations and consistent within contiguous training folds; every fitting partition contains both classes. Explicit valid seed, hidden size, epochs, learning rate, dropout and label smoothing. Return detached immutable JSON configuration. Caller owns image meaning, units and group identity; no file decoding or duplicate discovery.',
        dependencies=[]),
    'fit': dict(
        plain='Train fold-local image and metadata features with a retained neural ensemble.',
        bounds='Revalidated Prepared configuration. Extract 31 fixed RGB moments, histograms, gradients and aspect descriptors. Each fitting fold learns numeric medians, categorical vocabulary and fused standardization only from its training rows. All-missing numeric columns use zero; unknown categories have zero indicator vectors. Train CPU float64 MLPs with seeded dropout, smoothed binary cross entropy, Adam and every configured epoch. Complete out-of-fold predictions and equal mean of retained fold probabilities. No learned image backbone, early stopping, OOF stacking or full-data refit. Dense features and full-batch training; trusted in-process models, no portable persistence or large-scale resource claim.',
        dependencies=[]),
    'calibrate': dict(
        plain='Choose an F1 decision threshold using separate calibration groups.',
        bounds='Unmodified Fitted ensemble and revalidated source configuration. Predict calibration probabilities without refitting any feature transform or neural model. Existing generic float64 empirical-F1 provider consumes one score column and int64 binary labels; both classes required. Inclusive threshold comparison, highest observed threshold among equal computed F1 values. Return threshold and calibration count while retaining the fitted ensemble. Decision selection is not probability calibration or independent accuracy; caller owns meaningful group separation.',
        dependencies=['fit_empirical_f1_thresholds']),
    'predict': dict(
        plain='Predict query images with metadata and apply the selected threshold without retraining.',
        bounds='Unmodified Calibrated state and revalidated configuration. Equal mean of retained fold probabilities in query order, then existing generic inclusive column-threshold provider. Return finite JSON scores, classes, threshold, population counts, fold and OOF counts, visual feature count, epochs and mean initial/final training loss. Query labels prohibited; no refit. Generic binary output, no named submission format, predictive utility guarantee or portable fitted state.',
        dependencies=['apply_column_thresholds']),
}

def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    if raw != subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT):
        raise ValueError('Immutable runtime evidence changed')
    evidence = json.loads(raw)
    if evidence['tests'] != dict(tests=63, failures=0, errors=0, skipped=0) or evidence['passed'] is not True:
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
    return dict(format='image_tabular-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        catalog_mutations=0, catalog_state_checked=False, review_source='automated',
        runtime_evidence_commit=EVIDENCE_COMMIT, runtime_evidence_sha256=sha(ROOT/EVIDENCE),
        reviewer_sha256=sha(__file__), providers=providers,
        reused_numeric_providers=[r for r in records.values() if r["role"] == "numeric"],
        limits=['Offline semantic candidate review, not catalog approval or a human Tier 1 designation.',
            'Two existing generic F1 providers are reused unchanged. Image meaning, units, grouping, descriptors, metadata transforms, models and ensemble policy remain explicit recipe choices.',
            '63 synthetic tests cover visual features, fold isolation, neural loss and dropout, native execution, F1 oracles and no-refit parity; no empirical deployment guarantee.',
            'Live catalog reconciliation, original source-scope review, transaction checks and served selection remain required.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/image_tabular_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), catalog_state_checked=False, approved=False)))

"""Offline semantic contracts bound to the complete temporal_selected runtime evidence."""
import inspect
import json
import subprocess

from scripts.validate_temporal_selected_complete_runtime import ROOT, sha, source_inventory
from scripts.review_video_numeric_runtime import review as runtime_review
from sciona.atoms.ml import temporal_sparse_selected_execution as adapters
from scripts.validate_temporal_selected_complete_runtime import PROVIDERS

EVIDENCE_COMMIT = '76b98767'
EVIDENCE = 'docs/reviews/temporal_selected_complete_runtime.json'
CONTRACTS = {
    'initialize': dict(
        plain='Start a metric-independent policy for selecting checkpoints and deciding when to stop.',
        bounds='Positive Python integer patience, finite nonnegative absolute min_delta and mode min or max. Return fresh immutable state with no best score and zero evaluations. Strict improvements larger than min_delta reset patience; ties retain earlier checkpoint. Metric computation, model storage, evaluation cadence and independent assessment remain caller responsibilities.',
        dependencies=[]),
    'observe': dict(
        plain='Record one finite metric and return a new checkpoint decision state.',
        bounds='Trusted immutable ValidationStopping state and finite Python numeric metric excluding bool. Return new state with zero-based best_index, best score, evaluation count and consecutive nonimprovement count. Reject observations after patience exhaustion. Validate constructor counters and checkpoint consistency. No model mutation, automatic checkpoint saving, relative improvement or accuracy guarantee. Finite floating-point arithmetic and strict threshold comparisons apply.',
        dependencies=[]),
    'prepare': dict(
        plain='Inspect private ordered streams and check resource limits, temporal separation and stopping controls.',
        bounds='Exact version-1 sparse stream recipe plus explicit max_epochs, patience and min_delta. Private JSONL training, validation and unlabeled query roles, finite records, globally nondecreasing time, positive line/source/output limits and bounded entity/history/pending capacities. Nonnegative targets. Strict train-validation-query temporal gaps. Hash width, history window, period, chunk size, seed, learning rate, penalty, feature clip and maximum prediction are explicit. Inspect descriptors without fitting; prepared paths and digests must remain private. Time units and target availability semantics are caller policy.',
        dependencies=[]),
    'execute': dict(
        plain='Train from verified snapshots, select a validation checkpoint, and predict future rows.',
        bounds='Trusted Prepared state from the selected recipe. Revalidate configuration, descriptors, all three immutable disk snapshots, row summaries, gaps and output limit before fitting. Replay training snapshots with fresh causal history each epoch; strict-past labels only, same-time targets deferred. Eight numeric features plus fixed-width unsigned hashed categories in bounded CSR chunks. Squared-error and Huber SGD on log1p targets, equal log-score mean, nonnegative capped inverse transform. Validation labels select epochs through reusable initialize/observe providers but never update training history or estimators. Retain earliest sufficiently improved checkpoint under absolute threshold and patience, bounded by max_epochs. Return ordered bounded query predictions, counts, selected epoch, scores and selection flag; exclude private paths/digests. Scores are selection evidence, not independent accuracy. No full-population feature materialization, portable model persistence, throughput guarantee or historical winner claim.',
        dependencies=['initialize', 'observe']),
}

def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    if raw != subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT):
        raise ValueError('Immutable runtime evidence changed')
    evidence = json.loads(raw)
    if evidence['tests'] != dict(tests=76, failures=0, errors=0, skipped=0) or evidence['passed'] is not True:
        raise ValueError('Complete passing runtime qualification required')
    for path, digest in evidence['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Qualified source changed: '+path)
    if source_inventory() != evidence['providers'] or runtime_review() != evidence['runtime']:
        raise ValueError('Registered source or runtime changed')
    records = {r['runtime_fqdn'].rsplit('.', 1)[1]:r for r in evidence['providers']}
    if set(records) != set(CONTRACTS):
        raise ValueError('Every provider requires semantic review')
    functions = {f.__name__:f for f in PROVIDERS.values()}
    functions.update({name:getattr(adapters, name) for name in ('prepare','execute')})
    providers = []
    for name, r in sorted(records.items()):
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
    return dict(format='temporal_selected-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        catalog_mutations=0, catalog_state_checked=False, review_source='automated',
        runtime_evidence_commit=EVIDENCE_COMMIT, runtime_evidence_sha256=sha(ROOT/EVIDENCE),
        reviewer_sha256=sha(__file__), providers=providers,
        limits=['Offline semantic candidate review, not catalog approval or a human Tier 1 designation.',
            'Two generic stopping operations remain independently reusable; metric, model, time semantics, hashing and clipping remain recipe policy.',
            '76 synthetic tests include stopping transitions, native execution, immutable snapshot replay, causal isolation and checkpoint restoration; no empirical deployment guarantee.',
            'Live catalog reconciliation, original source-scope review, transaction checks and served selection remain required.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/temporal_selected_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), catalog_state_checked=False, approved=False)))

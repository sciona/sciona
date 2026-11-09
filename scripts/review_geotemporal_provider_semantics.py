"""Offline semantic contracts bound to the complete geotemporal runtime evidence."""
import inspect
import json
import subprocess

from scripts.validate_geotemporal_complete_runtime import ROOT, sha, source_inventory
from scripts.review_video_numeric_runtime import review as runtime_review
from sciona.geotemporal_complete_graph import adapters
from scripts.validate_geotemporal_complete_runtime import PROVIDERS

EVIDENCE_COMMIT = '5814ddea'
EVIDENCE = 'docs/reviews/geotemporal_complete_runtime.json'
CONTRACTS = {
    'blend_neighbor_means': dict(
        plain='Blend each numeric value with the mean of its explicitly selected neighbors.',
        bounds='Nonempty finite real vector, aligned square boolean neighbor mask and Python scalar strength in [0,1], excluding bool. Rows select neighbors; self-selection allowed. Empty rows retain their original values. Fresh finite float64 vector, signed values allowed. Adjacency, units, clipping and domain meaning are caller policy. Directed or unequal-degree masks need not conserve the global sum. Dense quadratic mask memory; no large-scale qualification. Reject unrepresentable arithmetic.',
        dependencies=[]),
    'prepare': dict(
        plain='Check shared coordinate and time units, observations, forward query times and model settings.',
        bounds='Exact finite version-1 JSON recipe with observations, context, contiguous forward blocks, query and controls. Caller declares shared planar coordinates in metres and seconds clock with shared origin; no coordinate transformation. Nonnegative observed targets; query labels rejected; unique site/time per population. Strictly ordered blocks, explicit gap and future query times. Radius, lookback, period, neighbor count, seed, trees, depth, leaf size and smoothing are explicit. Returns detached immutable JSON configuration.',
        dependencies=[]),
    'features': dict(
        plain='Join available context and past targets while building separate features for each validation fold.',
        bounds='Prepared configuration from qualified prepare, revalidated. For each site, choose latest context within lookback at or before each feature time; take nearest sites within metric radius with deterministic coordinate tie-breaking. Strict-past target lags use only the corresponding fold training history. Eleven features: coordinates, seasonal sine/cosine, context mean/distance/age/presence, historical target mean/presence/count. Missing aggregates are zero with presence flags. Build final training and query features using full observed history. Context timestamps are treated as availability times; caller must ensure this is valid. Raw coordinates remain features; no unseen-location or spatial-holdout claim.',
        dependencies=[]),
    'fit': dict(
        plain='Fit forward validation models, measure held-out errors, and refit a final model on all training rows.',
        bounds='Unmodified Featured state from qualified features. ExtraTreesRegressor with explicit seeded controls and n_jobs=1. Each expanding fold trains only its training targets; held-out predictions are clipped then blended only among same-time held-out neighbors using the generic numeric provider. Warmup block has no held-out predictions; report MSE on validated rows only. Final model refits all observed training rows. Query rows and labels do not train or select the model. Forward validation permits repeated locations and does not prove spatial independence. Trusted in-process arrays and estimators; forged or concurrent mutation excluded.',
        dependencies=['blend_neighbor_means']),
    'predict': dict(
        plain='Predict the requested future rows using the final fitted model without training again.',
        bounds='Unmodified Fitted state from qualified fit. Apply the full-training estimator to the already built causal query features; require finite aligned predictions. Preserve query input order. Return immutable prediction tuple with fitted context. No refitting or estimator mutation. Coordinate/clock and future-query gap policy remain those of preparation; arbitrary new query schemas and portable state persistence are not qualified.',
        dependencies=[]),
    'finish': dict(
        plain='Apply the chosen clipping and neighbor smoothing rules and return predictions with validation summaries.',
        bounds='Unmodified Predicted state from qualified predict. Clip to nonnegative values, construct a boolean mask excluding self and selecting exact same-time queries within the explicit planar radius, and call generic neighborhood blending with configured strength. Return finite JSON predictions, training/query/validated/warmup row counts, fold/tree counts, feature count and forward-validation MSE. Query population affects smoothing. No conservation, map matching, named submission format, independent final accuracy or unseen-location accuracy guarantee.',
        dependencies=['blend_neighbor_means']),
}

def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    if raw != subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT):
        raise ValueError('Immutable runtime evidence changed')
    evidence = json.loads(raw)
    if evidence['tests'] != dict(tests=65, failures=0, errors=0, skipped=0) or evidence['passed'] is not True:
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
    functions.update({name:getattr(adapters, name) for name in ('prepare','features','fit','predict','finish')})
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
    return dict(format='geotemporal-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        catalog_mutations=0, catalog_state_checked=False, review_source='automated',
        runtime_evidence_commit=EVIDENCE_COMMIT, runtime_evidence_sha256=sha(ROOT/EVIDENCE),
        reviewer_sha256=sha(__file__), providers=providers,
        limits=['Offline semantic candidate review, not catalog approval or a human Tier 1 designation.',
            'One numeric neighborhood operation remains independently reusable; units, adjacency, populations and model choices remain recipe policy.',
            '65 synthetic tests include numeric oracles, native execution and retained-recipe parity; no empirical deployment guarantee.',
            'Live catalog reconciliation, original source-scope review, transaction checks and served selection remain required.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/geotemporal_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), catalog_state_checked=False, approved=False)))

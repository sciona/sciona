"""Map the exact original intake to the qualified validation-selected sparse temporal realization."""
import hashlib
import json
import subprocess

from sciona.competition_import import FORMAT, _canonical, _id
from scripts.review_text_pair_original_scope import anchor
from scripts.review_temporal_selected_provider_semantics import review as provider_review
from scripts.validate_temporal_selected_complete_runtime import ROOT, sha

ORIGINAL = 'ad8e992f-1e5d-530c-9475-c9caf4a08bad'
ORIGINAL_HASH = '5ee3659042611115b7a83dbf165f7b44355a0f09ff346514ce4dd2a15e66340b'
IDENTITY = '5acf5fba-a858-5d4a-9832-77b99bea6714'
STAGES = {
    'efficient_table_loading': dict(
        anchors=[('temporal_sparse_source','inspect_source'), ('temporal_sparse_source','snapshot'), ('temporal_sparse_selected_execution','verified_streams')],
        realization='Select bounded chunked JSONL event streams. Inspect labels, order and limits, then verify immutable disk snapshots of all populations before fitting. Replay snapshots without copying full populations into RAM.',
        limits='Caller supplies the exact runtime event interface and private paths. No HDF5, Parquet, arbitrary table schema or measured high-volume throughput claim. Snapshot disk space is bounded by configured source sizes; three snapshots coexist.'),
    'temporal_feature_engineering': dict(
        anchors=[('temporal_sparse_features','TemporalSparseEncoder._advance'), ('temporal_sparse_features','TemporalSparseEncoder.encode')],
        realization='Compute value, periodic sine/cosine, per-entity rolling and latest-time target means, history age/count and presence. Defer same-time targets across chunk boundaries. Rebuild encoder from empty history each epoch and preserve row order.',
        limits='Targets are available immediately after observation time; delayed availability unsupported. Caller specifies units, period and window. Retained entity keys and pending/history capacities fail closed on exhaustion. No arbitrary calendar or market-specific features.'),
    'categorical_sparse_encoding': dict(
        anchors=[('temporal_sparse_features','TemporalSparseEncoder._matrix'), ('temporal_sparse_training','clipped')],
        realization='Select fixed-width unsigned entity/category hashing and eight numeric features, assembled as CSR chunks and clipped using an explicit fixed bound. No target statistics are fit from validation rows.',
        limits='Hash collisions are intentional. No native CatBoost categoricals, learned target encoding or vocabulary branch. No dense population conversion; per-chunk memory still depends on explicit hash width and chunk size.'),
    'leakage_safe_validation': dict(
        anchors=[('temporal_sparse_source','validate_time_splits'), ('temporal_sparse_selected_execution','verified_streams'), ('temporal_sparse_training','StreamEnsemble.predict_batches')],
        realization='Require strict chronological train-validation-query separation with explicit gaps before fitting. Validation/query labels never update history or model coefficients. Use validation RMSLE only for epoch selection.',
        limits='Entities may repeat; this is temporal rather than entity-disjoint validation. Reusing validation for stopping is selection, not independent final evaluation. No rolling refits, group holdout or validation-label history updates.'),
    'boosted_sparse_ensemble': dict(
        anchors=[('temporal_sparse_training','fit'), ('temporal_sparse_selection','fit_selected'), ('validation_stopping','ValidationStopping.observe')],
        realization='Select the permitted sparse linear ensemble branch: squared-error and Huber SGD on log1p targets with equal log-score averaging. Train successive causal stream passes, evaluate each epoch, and restore the earliest sufficiently improved checkpoint using registered generic stopping providers and explicit patience, threshold and epoch budget.',
        limits='Model controls and training chunk size affect results. Validation selects epoch only; no hyperparameter search, boosted-tree branch or refit on validation labels. No historical winner reproduction or empirical accuracy guarantee.'),
    'metric_postprocessing': dict(
        anchors=[('temporal_sparse_training','StreamEnsemble.predict_batches'), ('temporal_sparse_training','StreamEnsemble.evaluate'), ('temporal_sparse_selected_execution','execute')],
        realization='Clip averaged log scores between zero and log1p(max_prediction), invert with expm1, and return ordered finite nonnegative capped predictions plus explicitly labeled selection diagnostics. Bound query output rows before fitting and during prediction.',
        limits='Generic structured output under the explicit recipe, not a named submission file. No rounding, zero-sum, calibration, classification threshold or online refit branch. Private source paths, digests and records are excluded from results.'),
}

def review():
    semantics_path = 'docs/reviews/temporal_selected_provider_semantics.json'
    raw = (ROOT/semantics_path).read_bytes()
    semantics = json.loads(raw)
    if semantics != provider_review():
        raise ValueError('Qualified provider semantics changed')
    directory = ROOT.parent/'sciona-atoms/data/solution_cdgs'
    name = 'large_scale_temporal_tabular_ensemble_topology'
    snapshot = dict(format=FORMAT, template=json.loads((directory/(name+'.json')).read_text()),
        bindings=json.loads((directory/(name+'_bindings.json')).read_text()))
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
    if (digest, _id(fqdn+'/'+digest), _id(fqdn)) != (ORIGINAL_HASH, ORIGINAL, IDENTITY):
        raise ValueError('Local source does not reconstruct the exact original intake')
    stages = snapshot['template']['stages']
    if len(stages) != len(STAGES) or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Every original stage must be covered exactly once')
    return dict(format='temporal_selected-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, catalog_state_checked=False, source_snapshot_reconstructed=True,
        source_version_id=ORIGINAL, source_content_hash=ORIGINAL_HASH, artifact_id=IDENTITY, fqdn=fqdn,
        provider_semantics_sha256=sha(ROOT/semantics_path), reviewer_sha256=sha(__file__),
        source_stage_review=[dict(source_stage={k:s[k] for k in ('stage_id','description','guarantees')},
            realization=STAGES[s['stage_id']]['realization'], limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision='Two independently reusable immutable stopping providers with two bounded-stream recipe adapters. Six source stages are realized inside one verified snapshot lifetime to avoid full-population materialization. Metric, hashing, capacities, time semantics and model choices remain explicit recipe policy.',
        remaining_requirements=['Recheck original snapshot and preserved derivative against live catalog state.',
            'Reconcile and approve exact provider versions with metadata, dependency and rollback checks.',
            'Bind, store and execute the corrected original, then verify Tier 3 publication and served retrieval.'],
        limits=['Source reconstruction proves exact intake content identity, not current database status or approval.',
            'All six source stages covered for the selected sparse temporal realization, including early stopping; temporal separation does not imply entity-disjoint or independent final evaluation.',
            'Automated synthetic Tier 3 candidate evidence only; no human Tier 1 designation or empirical deployment guarantee.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/temporal_selected_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(result['source_stage_review']), exact_source=True, approved=False)))

"""Automated Tier 3 source-scope review for the generic Future Sales topology."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='e6803752-90ef-5cab-8b30-5ebeec716c15'
SOURCE_HASH='ffde31b5d874a8d6fafa0103988b0b9a8e0f13a0169f39ff6dce4e30369423e2'
SUFFIXES=['source_triage','runtime_tests','graph_execution','worker_execution','environment']
LIMITATIONS=[
 'Automated Tier 3 Community generic benchmark reconstruction, not a historical winning solution. Tier 1 human certification and Tier 2 usage qualification are not claimed. Original intake remains draft with mandatory provenance.',
 'Caller supplies private integer day ordinals, strictly increasing period boundaries and a fixed entity universe. Missing observed-period transactions mean zero. This assumes entities are active throughout the supplied history; no inferred opening/closing dates or source calendar parsing.',
 'Quantities are netted within each period/entity before clipping to zero through twenty, following the host target clarification. Negative returns are allowed before aggregation. Final-period targets remain unobserved.',
 'Lags, complete-width shifted rolling means and entity/store/item/category/global expanding means use prior clipped targets only. Explicit cold-start mean and cyclic period features; no inferred holiday/promotion inputs or learned embeddings.',
 'The generic LightGBM branch uses squared-error training and clipped-prediction RMSE validation. Native categorical entity slots, deterministic single-thread tree construction and explicit seed/search bounds. XGBoost and optional ensemble blending are not claimed.',
 'TPE search must include model-based trials beyond random startup. Early stopping selects iterations using a rolling one-step validation population; final model refits observed periods from the configured start at the selected round count.',
 'Earlier validation-period observations may inform later validation features. This is not fixed-origin multi-period forecasting. The optimization score selects hyperparameters and is not an unbiased held-out generalization estimate.',
 'The forecast covers exactly the final supplied period and preserves entity input order, with predictions clipped to zero through twenty. No historical accuracy or raw-to-final competition file equivalence is claimed.',
 'In-process execution in the combined graph runner reproducibly segfaulted after imports. Exact native cause is unestablished. The supported graph executes the same numerical lifecycle in a fresh process using the current reviewed virtual-environment interpreter.',
 'Private payloads/results use JSON pipes without payload files. Worker diagnostics are captured and failures reported generically. Default worker timeout is 3600 seconds, configurable by SCIONA_FUTURE_SALES_TIMEOUT_SECONDS. Dedicated worker execution is required.',
 'Synthetic evidence covers temporal feature invariance, aggregation/clipping, TPE model-based sampling, deterministic search/refit, isolated/direct agreement, failure/timeout guards and actual graph execution. No empirical accuracy guarantee.',
 'Pinned dependency closure and retained LightGBM/Optuna notices qualify the provisioned Python 3.13 macOS arm64 worker. Clean installation, other platforms, resource caps, trained-model interchange and interrupted-run resume are not qualified.',
 'No benchmark records, entity identities, label inventories or private templates were downloaded or embedded. Actual inputs, forecasts and validation metrics remain private runtime values.'
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        p=(base/name).resolve();assert p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for semantic review')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_future_sales_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH
    assert source['source_scope']=='Generic benchmark topology with competition landing-page reference; no historical winning implementation identified.'
    for name in SUFFIXES[1:]:assert docs[name]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==29 and tests['serialized_boundary_tests']==12
    for k in ['aggregation_before_clipping','current_and_future_target_feature_invariance','period_prefix_invariance','complete_tpe_early_stopping_refit','model_based_tpe_sampling_exercised','deterministic_repeated_prediction','isolated_worker_matches_direct','worker_failure_timeout_and_response_guards']:assert tests[k]
    check_hashes(ROOT,tests['hashes'])
    graph=docs['graph_execution'];assert graph['checks']==dict(actual_runner_nodes=2,training_rows=40,validation_rows=16,refit_rows=56,forecast_rows=8,trials=3,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True)
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('future_sales_*.py')}<=set(graph['code_sha256'])
    check_hashes(ROOT,graph['code_sha256'])
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/future_sales_execution.py';assert sha(provider)==graph['provider_sha256']
    from sciona.future_sales_graph import build_future_sales_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_future_sales_graph())[0]==graph['serialized_graph_sha256']
    worker=docs['worker_execution'];assert worker['isolated_full_graph_passed'] and worker['isolated_matches_direct'] and worker['in_process_graph_exit_code']==139
    check_hashes(ROOT,worker['hashes'])
    env=docs['environment'];requirements=ROOT/'requirements/future-sales-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert {'numpy','scipy','lightgbm','optuna'}<=set(dependencies) and env['declared_dependency_closure_compatible']
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'FutureSales-lightgbm-LICENSE.txt','FutureSales-optuna-LICENSE.txt','FutureSales-optuna-LICENSE_THIRD_PARTY.txt'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_future_sales_execution.py','requirements/future-sales-execution.txt']+['docs/licenses/'+n for n in env['license_sha256']]
    return dict(format='future_sales-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,
        source_scope=source['source_scope'],serialized_graph_sha256=graph['serialized_graph_sha256'],provider_sha256=sha(provider),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),
        intake_stage_mapping=[dict(intake=a,realization=b) for a,b in [('time_aggregation','Explicit entity/period net quantity aggregation'),('target_clipping','Clip only after aggregation'),('lag_feature_generation','Strict prior lag and rolling means'),('mean_encoding','Prior expanding group means and explicit cyclical features'),('gbdt_modeling','LightGBM/TPE search, clipped-RMSE early stopping and final observed-period refit')]],
        dependencies=dependencies,limitations=LIMITATIONS,evidence_sha256={f'competition_future_sales_{s}.json':sha(reviews/f'competition_future_sales_{s}.json') for s in SUFFIXES},auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_future_sales_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3,evidence=len(SUFFIXES))))

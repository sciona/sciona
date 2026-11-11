"""Automated Tier 3 source-scope review for the generic March topology."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='26d177bb-7d6d-5ee1-b881-2bafe69064cd'
SOURCE_HASH='a32fed30707dd363c95f39462a6bd6db505102f4449f3dccf24fb62d71af34ee'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=[
 'Automated Tier 3 Community generic implementation. No Tier 1 human certification or Tier 2 usage qualification. Original intake remains draft with mandatory provenance.',
 'The intake supplies only a generic five-stage benchmark and a Kaggle homepage reference. There is no identified historical winning source or performance to certify. This is an explicit new realization of the logistic-regression branch, not XGBoost or historical winner parity.',
 'Caller supplies private nonnegative integer box totals, team slots, season identifiers and unique within-season game order. Completed games must have unequal points and positive estimated possessions. No source file parsing, resampling, missing-value imputation or official box-score provenance validation is included.',
 'Estimated possessions equal field-goal attempts plus explicit free-throw coefficient times free-throw attempts minus offensive rebounds plus turnovers. Offensive/defensive efficiency and turnover rates use ratios of summed totals.',
 'PageRank transfers rank from loser to winner, one weighted edge unit per game. Explicit damping, tolerance and iteration limit; uniform default personalization/dangling behavior. This graph feature is not a certified true strength-of-schedule measure.',
 'Elo resets each season with explicit initial value, K and scale. Games are processed in explicit order, with neutral-court win/loss updates. Margin and home advantage are excluded; no historical rating-parameter equivalence is claimed.',
 'Features concatenate Team A, Team B and their difference. StandardScaler and lbfgs logistic regression fit only training seasons. Mirrored matchups augment fitting and calibration, with reversed outcomes.',
 'Training, calibration and prediction use strictly disjoint ordered seasons; all supplied regular-season games must precede each target matchup in that season. This supports postseason predictions, not rolling in-season estimation. Caller remains responsible for genuine temporal/source provenance.',
 'A FrozenEstimator preserves the fitted base; sigmoid calibration uses one explicit held-out population and does not refit the base. Final forward/reverse probability projection enforces orientation symmetry. It is not an empirical calibration guarantee.',
 'Synthetic checks cover direct efficiency/Elo calculation, Elo conservation, independent PageRank linear-system agreement, chronology and identity guards, future-season independence, calibration isolation and actual serialized graph execution. No real-world generalization or competition accuracy claim.',
 'Qualified on the provisioned Python 3.13 macOS arm64 worker with pinned sklearn/NumPy/SciPy/NetworkX dependencies and retained notices. Clean installation, other platforms, resource limits, saved-model interchange and process resume are not qualified.',
 'Private input records and probabilities remain runtime values. Evidence uses generated numeric games only; no competition records, team identities, labels or historical outputs are embedded.'
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        p=(base/name).resolve();assert p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for semantic review')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_march_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH
    assert source['source_scope']=='Generic five-stage benchmark topology; intake supplies only the Kaggle homepage, no historical solution or source implementation.'
    for name in SUFFIXES[1:]:assert docs[name]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==22 and tests['serialized_boundary_tests']==7
    for k in ['complete_feature_fit_calibration_prediction','independent_pagerank_linear_system','elo_conservation','current_season_feature_order_invariance','future_season_feature_independence','calibration_does_not_refit_base']:assert tests[k]
    check_hashes(ROOT,tests['hashes'])
    graph=docs['graph_execution'];assert graph['checks']==dict(actual_runner_nodes=2,training_rows=4,calibration_rows=4,predicted_matchups=2,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True)
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('march_*.py')}<=set(graph['code_sha256'])
    check_hashes(ROOT,graph['code_sha256'])
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/march_execution.py';assert sha(provider)==graph['provider_sha256']
    from sciona.march_graph import build_march_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_march_graph())[0]==graph['serialized_graph_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/march-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['direct_runtime_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'numpy','scipy','scikit-learn','networkx','joblib','threadpoolctl'}
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'March-sklearn-BSD.txt','March-NetworkX-BSD.txt'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_march_execution.py','requirements/march-execution.txt']+['docs/licenses/'+n for n in env['license_sha256']]
    return dict(format='march-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,
        source_scope=source['source_scope'],serialized_graph_sha256=graph['serialized_graph_sha256'],provider_sha256=sha(provider),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),
        intake_stage_mapping=[dict(intake=a,realization=b) for a,b in [('season_aggregation','Explicit box-score efficiency/turnover aggregation'),('graph_centralities','Weighted loser-to-winner PageRank'),('elo_ratings','Ordered neutral-court per-season Elo'),('pairwise_concatenation','Both team vectors and their difference'),('logistic_regression','Training-only scaler/logistic fit, held-out sigmoid calibration and orientation projection')]],
        dependencies=dependencies,limitations=LIMITATIONS,evidence_sha256={f'competition_march_{s}.json':sha(reviews/f'competition_march_{s}.json') for s in SUFFIXES},auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_march_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3,evidence=len(SUFFIXES))))

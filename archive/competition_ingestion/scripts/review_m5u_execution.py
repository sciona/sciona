"""Read-only automated Tier 3 review of the corrected uncertainty execution."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
from scripts.validate_m5u_environment import dependency_closure
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='5d2f953e-86bb-57f8-9a97-e0b6a303fd94'
SOURCE_HASH='69deb6e73d3e9e16de3fa53ef108b6a2928c0bed6706bf72ba5c8e558a97d8a5'
SCOPE='Independent corrected direct-quantile uncertainty pipeline with all 14 levels and 28-day restored forecasts'
STAGE_MAPPING={
    'base_forecasting':'Hierarchy preparation and direct quantile regressors replace the unsupported intake point-forecast dependency',
    'residual_calculation':'Residual-distribution stage excluded; targets and features use expanding absolute-difference volatility scaling',
    'probability_distribution_fitting':'Gaussian and negative-binomial fitting claims excluded; grouped LightGBM quantile objective is used directly',
    'quantile_generation':'Nine source quantiles, quantile-weighted sampling, per-level grouped search and repeated perturbed inference',
    'hierarchical_scaling':'Source level factors and median-total adjustment, with explicit retained-base-reference correction for undefined aggregate outlet denominators'}
LIMITATIONS=[
    'Automated Tier 3 Community only; no Tier 1 human review or Tier 2 usage qualification. Original intake remains draft with mandatory exact provenance.',
    'Independent reconstruction; original archive reuse grant remains unresolved. No original implementation, data, trained models or notebook outputs are embedded in this implementation.',
    'The intake point-forecast/residual/distribution recipe is contradicted by pinned U1 software and explicitly replaced in this corrected execution.',
    'Aggregate outlet normalization is an independent correction: preserve defined filtered-base means; otherwise use pre-filter base histories at the outlet, or the full base pool for aggregate outlet roles. Historical implementation parity is unproven.',
    'Independent deterministic CPU, single-thread, feature naming and per-stage RNG choices do not reproduce historical global/time seeds, learned weights or competitive scores.',
    'Full default schedule: 14 configured levels, nine quantiles, one bag and latest outer holdout; global controller controls quantile masks. Outer row exclusion is not an unbiased future-test guarantee.',
    'Full-budget synthetic graph execution covers 126 models, 4482 candidate fits and 2730000 repeated queries. Historical population scale, accuracy, cross-platform, clean-install and production qualification are unclaimed.',
    'Canonical finite JSON input is capped at 256 MiB. Caller supplies aligned integer hierarchy roles, three category partitions, prices and complete consecutive calendar/state roles. Each invocation retrains.',
    'Source cleaning excludes certain calendar rows and requires a retained final history day. Undefined sampling populations, nonpositive volume scales and nonfinite feature/prediction arithmetic reject.',
    'Forecasts remain unrounded, unsorted and unclipped; no quantile monotonicity guarantee. Source CSV rounding is a presentation step outside the numerical result.'
]
SUFFIXES=['execution_source_review','component_tests','loss_comparison','prediction_comparison','training_parameters',
          'normalization_correction','training_schedule','full_pipeline','graph_execution','environment']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_m5_uncertainty_{s}.json').read_text()) for s in SUFFIXES}
    for suffix,d in docs.items():
        assert d['approved'] is False and d['catalog_mutations']==0
        assert d['status']==('component_validated' if suffix=='normalization_correction' else 'passed')
    source=docs['execution_source_review']
    assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH
    assert source['scope']==SCOPE and source['stage_mapping']==STAGE_MAPPING
    pins_path=reviews/'competition_m5_uncertainty_source_pins.json';pins=json.loads(pins_path.read_text())
    assert sha(pins_path)==source['source_pins_sha256']
    assert pins['commit']=='18e4e776cfa1ded3b387fabe008c6c2c44922007' and len(pins['files'])==3
    for suffix in SUFFIXES[1:-2]:check_hashes(ROOT,docs[suffix]['sha256'])
    tests=docs['component_tests'];assert tests['checks']==dict(tests_passed=102,synthetic_only=True)
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_m5u_*.py')}<=set(tests['sha256'])
    assert docs['loss_comparison']['checks']==dict(synthetic_only=True,source_comparisons=400)
    assert docs['prediction_comparison']['checks']==dict(synthetic_only=True,forecast_comparisons=202,hierarchy_comparisons=100,native_quantiles=9,native_models=27,candidate_fits=54,refits=27)
    assert docs['training_parameters']['checks']==dict(source_schedule_comparisons=126,parameter_levels=14,quantiles=9,quantile_weights=9)
    for suffix in ('loss_comparison','prediction_comparison','training_parameters'):
        assert docs[suffix]['source_software_sha256'] in pins['files'].values()
    correction=docs['normalization_correction']
    assert correction['policy']=='retained_base_reference' and 'not proven' in correction['limitation']
    assert correction['checks']==['Defined source partition equivalence','Independent pandas daily mean oracle','Aggregate batch composition invariance','Prefix causality','Input immutability','Concrete outlet isolation','Default source mode still rejects undefined denominators']
    schedule=docs['training_schedule']
    assert schedule['checks']==dict(levels=14,models=126,quantiles=9,candidate_fits=4482,refits=126,full_source_search_counts=True,full_source_sampling_rates=True,outer_group_isolation=True)
    from sciona.m5u_training import PARAMETERS,schedule as plan
    assert {int(k) for k in schedule['levels']}==set(PARAMETERS)
    for key,row in schedule['levels'].items():
        assert row['iterations']==plan(int(key),1.)['iterations'] and row['models']==9 and row['outer_group_isolation']
        assert row['scale_range']==PARAMETERS[int(key)][1]
    full=docs['full_pipeline']
    assert full['checks']==dict(levels=14,models=126,quantiles=9,horizons=28,query_rows=2730000,candidate_fits=4482,refits=126,full_source_search_counts=True,full_source_inference_budget=True,complete_hierarchy_coverage=True,finite_restored_predictions=True,outer_group_isolation=True)
    assert full['normalization']=='retained_base_reference'
    graph_report=docs['graph_execution'];check_hashes(ROOT,graph_report['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('m5u_*.py')}<=set(graph_report['code_sha256'])
    assert 'sciona/m5u_search_parameters.json' in graph_report['code_sha256']
    assert graph_report['source_version_id']==SOURCE_VERSION and graph_report['source_content_hash']==SOURCE_HASH
    assert graph_report['checks']==dict(actual_runner_nodes=2,models=126,quantiles=9,horizon=28,query_rows=2730000,full_source_training_and_inference_budgets=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True)
    from sciona.m5u_graph import build_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.m5u_execution as provider
    graph=build_graph();assert encode_execution_graph(graph)[0]==graph_report['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        fn=getattr(provider,'m5u_'+node.node_id)
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs] and all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_m5u_'+node.node_id)(witness)
    assert witness=={'kind':'M5U.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==graph_report['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/m5u-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert len(dependencies)==12 and dependencies==env['dependency_versions']==dependency_closure()
    assert sha(requirements)==env['requirements_sha256'] and env['verifier_sha256']==sha(ROOT/'scripts/validate_m5u_environment.py')
    assert env['python_version']==platform.python_version() and env['platform']==platform.system() and env['machine']==platform.machine()
    assert env['execution_device']=='cpu' and env['actual_root_imports_verified'] is True and env['declared_dependency_constraints_satisfied'] is True
    assert len(env['license_sha256'])==36;check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_m5u_execution.py','scripts/validate_m5u_environment.py','requirements/m5u-execution.txt','docs/reviews/competition_m5_uncertainty_source_pins.json']+['docs/licenses/'+p for p in env['license_sha256']]
    return dict(format='m5u-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
                serialized_graph_sha256=graph_report['serialized_graph_sha256'],provider_sha256=sha(provider_path),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=STAGE_MAPPING,
                dependencies=dependencies,limitations=LIMITATIONS,evidence_sha256={f'competition_m5_uncertainty_{s}.json':sha(reviews/f'competition_m5_uncertainty_{s}.json') for s in SUFFIXES},auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_m5_uncertainty_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],tier=result['proposed_tier'],catalog_mutations=0)))

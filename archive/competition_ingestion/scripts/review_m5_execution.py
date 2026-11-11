"""Read-only automated Tier 3 review for the independent M5 execution."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
from scripts.validate_m5_environment import dependency_closure
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='f64bac1a-ca7f-5f29-a645-ec7b2b42ca4b'
SOURCE_HASH='8af8fd4a570aac38f64592903046d4b229fe13604410c786a54ddd9fccc9c949'
SCOPE='Independent source-corrected M5 six-family CPU pipeline with 28-day equal-mean forecast'
STAGE_MAPPING={'temporal_unrolling': 'Day-major history, future rows, release filtering and runtime calendar/price joins', 'recursive_feature_engineering': 'Source lag inventory, fixed-window sample moments, cutoff target encoding and daily temporary means', 'objective_function_optimization': 'Six pooled Tweedie families at full 3000 configured rounds with source settings and independent CPU choices', 'magic_multiplier': 'Unsupported intake claim excluded: inspected final ensemble uses equal arithmetic mean with no constant multiplier', 'hierarchical_reconciliation': 'Unsupported final-stage claim excluded: inspected final ensemble averages six aligned base-series outputs; no reconciliation is applied'}
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human review or Tier 2 usage qualification. Original intake remains draft with mandatory exact provenance.', 'Independent mathematical reconstruction; no original software, data, outputs or weights copied into implementation. Original archive reuse grant unresolved; pinned software used for component comparisons.', 'Explicit semantic hierarchy/category/calendar/price roles generalize the historical schema. Caller supplies complete integer roles, aligned calendar and source-appropriate ordering.', 'Intake constant multiplier and final reconciliation claims are excluded; implemented final operation is the equal mean of six aligned family forecasts.', 'Known future prices participate in group features. Target moments include eligible training labels; recursive diagnostics overlap training, and nonrecursive missing diagnostic labels become zero. No unbiased validation claim.', 'Source range-based precision, missing windows, sample deviations and first-calendar-week semantics retained. Invalid joins and role mappings reject rather than silently multiplying rows; float16 overflow rejects where documented.', 'Every supplied pool in all six families trains for 3000 evaluated rounds without early stopping. Source leaf constraints can produce fewer retained iterations; full synthetic inventory ranges from1to3000 retained iterations.', 'CPU single-thread deterministic column-wise LightGBM and in-memory transport are independent runtime choices. No historical hardware, learned-weight or RNG parity.', 'Recursive families keep separate100day histories and recompute temporary means daily. Quantized historical values and float64 prediction feedback generalize current pandas source promotion behavior.', '96 synthetic tests and full220model/28day serialized execution qualify computation on70syntheticseries and250historydays. Historical sample volume, predictive accuracy, cross-platform, clean-install and production behavior are unqualified.', 'Finite JSON input capped at256MiB; opaque identities are validated but duplicate contents under renamed identities are not detected. Every invocation retrains all families.']
SUFFIXES=['execution_source_review', 'component_tests', 'lag_comparison', 'encoding_comparison', 'precision_comparison', 'training_parameters', 'full_inventory', 'boundary_execution', 'graph_execution', 'environment']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        p=(base/name).resolve()
        assert p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_m5_{s}.json').read_text()) for s in SUFFIXES}
    for d in docs.values():assert d['approved'] is False and d['catalog_mutations']==0
    source=docs['execution_source_review'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH
    assert source['scope']==SCOPE and source['stage_mapping']==STAGE_MAPPING
    pins_path=reviews/'competition_m5_source_pins.json'
    pins=json.loads(pins_path.read_text())
    assert sha(pins_path)==source['source_pins_sha256']
    assert pins['commit']=='18e4e776cfa1ded3b387fabe008c6c2c44922007' and len(pins['files'])==15
    for suffix in SUFFIXES:assert docs[suffix]['status']=='passed'
    for suffix in SUFFIXES[1:-2]:check_hashes(ROOT,docs[suffix]['sha256'])
    tests=docs['component_tests'];assert tests['checks']==dict(tests_passed=96,synthetic_only=True)
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_m5_*.py')}<=set(tests['sha256'])
    for suffix in ('lag_comparison','encoding_comparison','precision_comparison','training_parameters'):
        assert docs[suffix]['source_commit']==pins['commit']
    for suffix in ('lag_comparison','encoding_comparison','precision_comparison'):
        assert docs[suffix]['source_software_sha256'] in pins['files'].values()
    assert docs['lag_comparison']['checks']==dict(source_function_comparisons=54,synthetic_only=True,interleaved_groups=True,missing_values=True,stored_float16=True,recursive_full_precision=True)
    assert docs['encoding_comparison']['checks']==dict(synthetic_only=True,source_columns_compared=22,exact_float16=True)
    assert docs['precision_comparison']['checks']==dict(synthetic_only=True,source_comparisons=139,exact_values_and_dtypes=True)
    training=docs['training_parameters'];assert training['checks']==dict(source_parameter_sets_compared=6,configured_rounds=3000)
    assert len(training['families'])==6
    assert {(f['recursive'],f['pooling']) for f in training['families']}=={(r,p) for r in (True,False) for p in ('outlet','outlet_category','outlet_department')}
    assert all(f['source_sha256'] in pins['files'].values() for f in training['families'])
    full=docs['full_inventory']
    assert full['checks']==dict(synthetic_only=True,model_families=6,models=220,family_model_counts=[10,30,70,10,30,70],configured_rounds_per_model=3000,evaluated_rounds_total=660000,horizon=28,synthetic_series=70,finite_forecast=True,exact_mean=True,code_unchanged=True)
    assert full['retained_iterations']==dict(minimum=1,maximum=3000)
    assert docs['boundary_execution']['checks']==dict(synthetic_only=True,models=220,horizon=28,synthetic_series=70,strict_json_output=True,identifiers_excluded=True,code_unchanged=True)
    graph_report=docs['graph_execution'];check_hashes(ROOT,graph_report['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('m5_*.py')}<=set(graph_report['code_sha256'])
    assert graph_report['source_version_id']==SOURCE_VERSION and graph_report['source_content_hash']==SOURCE_HASH
    assert graph_report['checks']==dict(actual_runner_nodes=2,models=220,raw_preprocessing=True,full_training_limits=True,synthetic_series=70,horizon=28,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True)
    from sciona.m5_graph import build_m5_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.m5_execution as provider
    graph=build_m5_graph();assert encode_execution_graph(graph)[0]==graph_report['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        fn=getattr(provider,'m5_'+node.node_id)
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs] and all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_m5_'+node.node_id)(witness)
    assert witness=={'kind':'M5.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==graph_report['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/m5-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert len(dependencies)==12 and dependencies==env['dependency_versions']==dependency_closure()
    assert sha(requirements)==env['requirements_sha256'] and env['verifier_sha256']==sha(ROOT/'scripts/validate_m5_environment.py')
    assert env['python_version']==platform.python_version() and env['platform']==platform.system() and env['machine']==platform.machine()
    assert env['execution_device']=='cpu' and env['actual_root_imports_verified'] is True and env['declared_dependency_constraints_satisfied'] is True
    assert len(env['license_sha256'])==36;check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_m5_execution.py','scripts/validate_m5_environment.py','requirements/m5-execution.txt','docs/reviews/competition_m5_source_pins.json']+['docs/licenses/'+p for p in env['license_sha256']]
    return dict(format='m5-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=graph_report['serialized_graph_sha256'],provider_sha256=sha(provider_path),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=STAGE_MAPPING,
        dependencies=dependencies,limitations=LIMITATIONS,evidence_sha256={f'competition_m5_{s}.json':sha(reviews/f'competition_m5_{s}.json') for s in SUFFIXES},auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_m5_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],tier=result['proposed_tier'],catalog_mutations=0)))

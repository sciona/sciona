"""Read-only automated Tier 3 review of the independent Porto ensemble."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
from scripts.validate_porto_environment import dependency_closure
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='ad3239bf-5a38-565a-81da-3d3cba9a02b0'
SOURCE_HASH='8b42f3175fa6408c91c325e24784554cc480637a33673bd7a2128e2e657b8dd8'
SCOPE='Independent Porto five-DAE-neural and one-LightGBM probability ensemble on the provisioned CPU runtime.'
STAGE_MAPPING={
 'missing_value_imputation':'Replace unestablished sentinel imputation with explicit dropped/categorical/binary roles; joint population one-hot encoding and RankGauss for DAE inputs. Numeric sentinels remain literal.',
 'denoising_autoencoder_features':'Train five explicitly configured DAEs on unlabeled training plus query populations, with same-column swap noise and clean MSE targets; extract caller-selected hidden activations.',
 'feature_concatenation':'Correct universal raw/latent concatenation: neural branches use hidden activations only; tree branch uses raw prepared features.',
 'model_training':'Five independent supervised neural classifiers plus one LightGBM; remove unestablished XGBoost. All configurations explicit; historical CV replay excluded.',
 'rank_average_ensemble':'Correct rank averaging to the equal arithmetic mean of six model probabilities.'}
SUFFIXES=['source_review','runtime_tests','execution_boundary','graph_execution','environment','recovered_dae_execution','recovered_branch_execution']
LIMITATIONS=[
 'Automated Tier 3 Community only; no human Tier 1 or usage-qualified Tier 2 designation. Exact original intake remains draft.',
 'Independent algorithm realization; no original private C++/CUDA code, data, weights or training logs copied. No historical accuracy or exact winning recipe claim.',
 'Only one historical neural configuration was recovered. Remaining neural configurations, tree settings, optimizer, fold schedule and checkpoint semantics unresolved; explicit caller configurations required.',
 'Joint training/query preprocessing and unlabeled DAE training are transductive. Numeric sentinels remain literal; caller supplies column roles.',
 'Average-tie midpoint RankGauss endpoints, self-eligible same-column swap donors, PyTorch initialization, SGD momentum and dropout scaling are independent choices.',
 'Each invocation trains five DAEs/classifiers plus one raw-prepared tree. No persistent serving, CV or calibration qualification.',
 '49 synthetic tests and actual serialized six-model execution use small independent configurations. One recovered-width branch separately executed full epochs with independent optimizer choices; not full historical ensemble replay.',
 'Provisioned CPU dependency closure and notices checked; no clean-install, GPU, cross-platform, large-population or production qualification.',
 'Finite private JSON capped at 64 MiB; disjoint opaque identities cannot detect duplicate contents under different identities. Detailed learner control values validate at each learner.'
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        p=(base/name).resolve()
        assert p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_porto_{s}.json').read_text()) for s in SUFFIXES}
    for d in docs.values():assert d['approved'] is False and d['catalog_mutations']==0
    source=docs['source_review'];assert source['source_scope']==SCOPE and source['classification']=='material_intake_correction_required'
    assert source['stage_mapping']==STAGE_MAPPING
    for suffix in ('source_review','runtime_tests','execution_boundary','graph_execution'):
        assert docs[suffix]['source_version_id']==SOURCE_VERSION and docs[suffix]['source_content_hash']==SOURCE_HASH
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==49 and tests['serialized_boundary_tests']==11
    check_hashes(ROOT,tests['sha256'])
    required={str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_porto_*.py')}
    assert required<=set(tests['sha256'])
    boundary=docs['execution_boundary'];assert boundary['tests_passed']==11;check_hashes(ROOT,boundary['sha256'])
    for suffix in ('recovered_dae_execution','recovered_branch_execution'):
        check_hashes(ROOT,docs[suffix]['sha256']);assert docs[suffix]['synthetic_only'] is True
    dae=docs['recovered_dae_execution'];branch=docs['recovered_branch_execution']
    expected=dict(hidden=[1500,1500,1500],feature_layers=[0,1,2],epochs=1000,batch_size=128,learning_rate=.003,decay=.995,swap_probability=.07,momentum=0.)
    assert dae['controls']==branch['dae_controls']==expected
    assert dae['epochs']==1000 and dae['extracted_features']==branch['learned_features']==4500
    assert 0<=dae['final_clean_mse']<dae['initial_clean_mse']
    assert branch['neural_controls']==dict(hidden=[1000,1000],epochs=200,batch_size=128,learning_rate=.0001,decay=.995,l2=.05,momentum=0.,dropout=.5,input_dropout=.1,dropout_scaling='inverted')
    execution=docs['graph_execution']
    assert execution['checks']==dict(actual_runner_nodes=2,models=6,dae_models=5,synthetic_query_rows=8,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True)
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('porto_*.py')}<=set(execution['code_sha256'])
    from sciona.porto_graph import build_porto_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.porto_execution as provider
    graph=build_porto_graph();assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        fn=getattr(provider,'porto_'+node.node_id)
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs] and all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_porto_'+node.node_id)(witness)
    assert witness=={'kind':'Porto.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']==boundary['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/porto-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert len(dependencies)==22 and dependencies==env['dependency_versions']==dependency_closure()
    assert env['requirements_sha256']==sha(requirements)
    assert env['python_version']==platform.python_version() and env['platform']==platform.system() and env['machine']==platform.machine()
    assert env['execution_device']=='cpu' and env['declared_dependency_constraints_satisfied'] is True and env['actual_root_imports_verified'] is True
    assert env['verifier_sha256']==sha(ROOT/'scripts/validate_porto_environment.py')
    assert len(env['license_sha256'])==61;check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_porto_execution.py','scripts/validate_porto_environment.py','requirements/porto-execution.txt']+['docs/licenses/'+p for p in env['license_sha256']]
    return dict(format='porto-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=STAGE_MAPPING,
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_porto_{s}.json':sha(reviews/f'competition_porto_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_porto_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verdict':result['verdict'],'tier':result['proposed_tier'],'catalog_mutations':0}))

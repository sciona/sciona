"""Read-only automated Tier 3 review for complete independent Amex execution."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
from scripts.validate_amex_environment import dependency_closure
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='d505faaa-7277-5559-b32c-f95c9f1f9d63'
SOURCE_HASH='f03e6e0c5dfb38e00baa98ea53f9a45a43b9ac4536c343bac2449ca09dbff138'
SCOPE='Independent full Amex raw preprocessing, manual/sequence features, grouped row DART, downstream tree/neural variants and literal blend on CPU.'
STAGE_MAPPING={
 'memory_optimization':'Correct generic memory optimization to explicit numeric flooring and runtime category mapping. Independent floating-point storage; no historical memory/downcast equivalence.',
 'time_series_aggregation':'Nine manual blocks, joint categorical and month ranks, timestamp-rank windows, row-model predictions, opposite padding directions and greedy normalized features.',
 'difference_features':'Ordered within-customer differences preserve missing adjacency; source mean/std/min/max/sum/last and floating floor division retained.',
 'model_training':'Five grouped row DART models plus two downstream five-fold DART and two five-fold neural variants. Remove unsupported XGBoost/CatBoost; preserve effective epochs, schedule, pooling and checkpoint metric.',
 'blending':'Literal ordered probability sum .30 manual tree + .35 tree with slots + .15 sequence neural + .10 combined neural; coefficients sum .90 with no normalization or rank averaging.'}
LIMITATIONS=[
 'Automated Tier 3 Community only; no Tier 1 human review or Tier 2 usage qualification. Exact original intake remains draft with mandatory provenance.',
 'Independent mathematical reconstruction; no original repository code, data, logs or model weights copied into repository. Specific original source reuse grant unresolved; privately pinned software used for numerical comparisons.',
 'Runtime numeric/category/month/time/fill roles generalize the source private schema. Caller must provide source-appropriate ordering, category maps and roles; no embedded private dataset or historical memory footprint claim.',
 'Joint train/query vocabularies, month ranks and binning are transductive. Row OOF predictions and downstream folds are not a nested unbiased validation pipeline.',
 'Source numeric flooring, average-tie timestamp windows, NaN aggregate semantics, compressed bin split omission, left-padded prediction slots and right-padded neural sequences retained. Undefined all-missing histograms and unsupported categories reject.',
 'Fifteen current CPU DART models use 4500 configured rounds; DART has no active early stopping. Single-thread deterministic column-wise mode and in-memory booster transport are independent choices.',
 'Ten CPU neural models use hidden width128, ten epochs, batch256 drop-last, raw BCE and effective Adam rates. Non-AMP default applies no label smoothing or gradient clipping. Strict positive Amex-metric checkpoint improvement required.',
 'Generalized neural input widths, CPU RNG consumption, in-memory checkpoint reload and single-worker batching differ from historical GPU execution. No native/GPU/RNG or learned-weight parity claim.',
 'Literal final blend coefficients sum .90. Output is a weighted score without normalization or calibration qualification.',
 '62 synthetic tests, pinned component comparisons, full learner controls and raw serialized 25-model execution establish computation; no historical competitive accuracy, production, cross-platform or clean-install qualification.',
 'Private finite JSON capped at64MiB; disjoint opaque identities do not detect duplicated content with renamed customers. Every invocation retrains the full lifecycle.'
]
SUFFIXES=['execution_source_review','component_tests','numeric_comparison','categorical_comparison','manual_comparison','sequence_comparison','binning_comparison','network_comparison','metric_comparison','row_training','tree_training','neural_training','pipeline','raw_execution','graph_execution','environment']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        p=(base/name).resolve();assert p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_amex_{s}.json').read_text()) for s in SUFFIXES}
    for d in docs.values():assert d['approved'] is False and d['catalog_mutations']==0
    source=docs['execution_source_review'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH
    assert source['source_scope']==SCOPE and source['stage_mapping']==STAGE_MAPPING
    pins=json.loads((reviews/'competition_amex_source_pins.json').read_text())
    assert pins['commit']=='03b86f584bb61a34e31eb6343bd33ae753a92c02' and len(pins['files'])==13
    assert source['source_pins_sha256']==sha(reviews/'competition_amex_source_pins.json')
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    for suffix in SUFFIXES[1:-2]:check_hashes(ROOT,docs[suffix]['sha256'])
    tests=docs['component_tests'];assert tests['tests_passed']==62
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_amex_*.py')}<=set(tests['sha256'])
    for suffix,count in [('numeric_comparison',24),('categorical_comparison',4),('metric_comparison',400)]:assert docs[suffix]['comparison_cases']==count
    for suffix,name in [('numeric_comparison','S2_manual_feature.py'),('categorical_comparison','S2_manual_feature.py'),('manual_comparison','S2_manual_feature.py'),('binning_comparison','S4_feature_combined.py'),('network_comparison','model.py'),('metric_comparison','utils.py')]:assert docs[suffix]['source_software_sha256']==pins['files'][name]
    assert docs['manual_comparison']['compared_blocks']==9
    assert docs['binning_comparison']['histogram_cases']==300 and docs['binning_comparison']['normalized_columns']==3
    sequence=docs['sequence_comparison'];assert sequence['source_software_pins']=={n:pins['files'][n] for n in ('S4_feature_combined.py','utils.py')}
    assert sequence['checks']==dict(joint_encoded_channels=True,prediction_left_padding=True,neural_right_padding=True,tabular_complement_only=True,sequence_lengths=[1,3,13])
    assert docs['network_comparison']['checks']==dict(network_cases=4,both_variants=True,source_input_dimensions=True,last_valid_timestep_pooling=True,input_gradients=True)
    for suffix,models in [('row_training',5),('tree_training',10)]:
        d=docs[suffix];assert d['models']==models and d['configured_rounds']==4500 and d['evaluated_rounds']==[4500]*models
    neural=docs['neural_training'];assert neural['models']==10 and neural['folds_per_variant']==5 and neural['epochs_per_fold']==10 and neural['hidden_width']==128 and neural['training_batch_size']==256 and neural['training_rows_per_fold_epoch']==512
    pipeline=docs['pipeline'];assert pipeline['models']==25 and pipeline['model_counts']==dict(row=5,downstream_tree=10,neural=10) and pipeline['actual_upstream_predictions'] is True and pipeline['score_kind']=='literal_weighted_sum_0.9'
    raw=docs['raw_execution'];assert raw['models']==25 and raw['raw_preprocessing'] is True and raw['full_training_controls'] is True and raw['strict_json'] is True
    execution=docs['graph_execution'];check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('amex_*.py')}<=set(execution['code_sha256'])
    assert execution['source_version_id']==SOURCE_VERSION and execution['source_content_hash']==SOURCE_HASH
    assert execution['checks']==dict(actual_runner_nodes=2,models=25,raw_preprocessing=True,full_training_limits=True,synthetic_query_rows=8,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True)
    from sciona.amex_graph import build_amex_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.amex_execution as provider
    graph=build_amex_graph();assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION];witness={}
    for node in graph.nodes:
        fn=getattr(provider,'amex_'+node.node_id)
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs] and all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_amex_'+node.node_id)(witness)
    assert witness=={'kind':'Amex.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/amex-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert len(dependencies)==22 and dependencies==env['dependency_versions']==dependency_closure()
    assert sha(requirements)==env['requirements_sha256'] and env['verifier_sha256']==sha(ROOT/'scripts/validate_amex_environment.py')
    assert env['python_version']==platform.python_version() and env['platform']==platform.system() and env['machine']==platform.machine()
    assert env['execution_device']=='cpu' and env['actual_root_imports_verified'] is True and env['declared_dependency_constraints_satisfied'] is True
    assert len(env['license_sha256'])==61;check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_amex_execution.py','scripts/validate_amex_environment.py','requirements/amex-execution.txt','docs/reviews/competition_amex_source_pins.json']+['docs/licenses/'+p for p in env['license_sha256']]
    return dict(format='amex-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=STAGE_MAPPING,
        dependencies=dependencies,limitations=LIMITATIONS,evidence_sha256={f'competition_amex_{s}.json':sha(reviews/f'competition_amex_{s}.json') for s in SUFFIXES},auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_amex_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(verdict=result['verdict'],tier=result['proposed_tier'],catalog_mutations=0)))

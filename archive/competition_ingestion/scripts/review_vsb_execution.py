"""Read-only automated Tier 3 review for the independent VSB execution."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
from scripts.validate_vsb_environment import dependency_closure
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='d79e40e4-45b0-5e74-a83c-c68c91f12afb'
SOURCE_HASH='565e511ee8d20ce0b32adf5c8cc726150f235b1f97d5502d62c96cfe339f5ac1'
SCOPE='Independent complete VSB recursive baseline, custom peaks, phase aggregation and repeated LightGBM signal decisions on CPU.'
STAGE_MAPPING={
 'wavelet_denoising':'Correct to recursive first-order baseline subtraction; no wavelet stage established.',
 'peak_detection':'Bidirectional directional scan intersection, sorted amplitude gradient and cumulative threshold knee with source negative slicing.',
 'peak_feature_extraction':'Four normalized waveform descriptors, clipped windows, source offset convention and aligned sawtooth MSE.',
 'signal_to_matrix_transform':'Correct to three-signal fundamental phase and nine measurement aggregates; no deep image transform established.',
 'tabular_model':'Full feature permutation and 25 repetitions of five LightGBM folds; both heldouts affect stopping.',
 'dl_model':'Remove unsupported deep branch; all 125 models are LightGBM.',
 'weighted_blend':'Correct to equal probability mean and signal-label MCC threshold from in-training scores; strict greater-than decisions broadcast over triples.'}
LIMITATIONS=[
 'Automated Tier 3 Community only; no human Tier 1 or usage-qualified Tier 2 designation. Original intake stays draft with mandatory exact source provenance.',
 'Independent mathematical reconstruction; no notebook source, data, outputs or weights copied into repository. Specific notebook reuse grant unresolved; private pinned source used only for numerical comparisons.',
 'Source corrections remove unsupported wavelet, image transform, deep branch and weighted tree/deep blend. Recursive baseline and custom peak edge semantics retained.',
 'Actual-length descriptor clipping and one-cycle FFT fundamental generalize the source fixed signal length. FFT summation order and near-boundary precision can differ; no native Numba parity.',
 'Missing adjacent samples and missing aggregate groups remain NaN. Undefined peak convolution, unaligned descriptor maximum and unusable random folds reject explicitly.',
 'Class-specific legacy random five-way assignments are not balanced K-fold partitions. Both validation and nominal test influence early stopping; neither provides untouched-test evaluation.',
 'Measurement labels are any-positive signal triples. Final threshold uses original signal labels against predictions from models trained on each measurement. Threshold optimizer uses >=; final decisions use strict >.',
 'Recovered 25 repetitions, 10000-round ceiling and 100-round stopping patience executed. Current single-thread deterministic LightGBM and callback API differ from historical runtime.',
 '57 synthetic tests, source component comparisons, full-limit aggregate training and full raw serialized execution establish computation only; no historical accuracy, GPU, clean-install, production or large-population qualification.',
 'Finite private JSON capped at 64 MiB; disjoint opaque identities do not detect duplicate contents under renamed identities. Every call retrains full ensemble.'
]
SUFFIXES=['execution_source_review','component_tests','source_numerics','peak_comparison','descriptor_comparison','measurement_comparison','threshold_comparison','full_training','raw_execution','graph_execution','environment']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        p=(base/name).resolve()
        assert p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_vsb_{s}.json').read_text()) for s in SUFFIXES}
    for d in docs.values():assert d['approved'] is False and d['catalog_mutations']==0
    source=docs['execution_source_review'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH
    assert source['source_scope']==SCOPE and source['stage_mapping']==STAGE_MAPPING
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    for suffix in SUFFIXES[1:-2]:check_hashes(ROOT,docs[suffix]['sha256'])
    tests=docs['component_tests'];assert tests['tests_passed']==57
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_vsb_*.py')}<=set(tests['sha256'])
    for suffix in ('source_numerics','peak_comparison','descriptor_comparison','measurement_comparison','threshold_comparison'):
        assert docs[suffix]['source_code_cells_sha256']=='ee289da71520cfa781723070f5face629c962e9430fd06bb4f2069a3b72a83b9'
    assert docs['source_numerics']['baseline_comparison_cases']==4
    assert docs['peak_comparison']['checks']==dict(exhaustive_directional_cases=9837,threshold_counter_cases=300,full_selection_cases=6)
    assert docs['descriptor_comparison']['checks']==dict(descriptor_cases=16,source_length_exercised=True,unaligned_maximum_rejected_by_both=True,missing_neighbor_nan_preserved=True)
    assert docs['measurement_comparison']['checks']==dict(aggregate_cases=3,aggregate_width=9,source_length_fourier_channels=3,phase_boundary_cases=4)
    assert docs['threshold_comparison']['comparison_cases']==402
    full=docs['full_training'];assert full['models']==125 and full['round_ceiling']==10000 and full['stopping_patience']==100
    assert full['checks']==dict(all_fold_features_and_labels_verified=True,both_holdouts_in_early_stopping=True,query_mean_oracle=True,signal_threshold_oracle=True,code_unchanged=True)
    assert 100<full['evaluated_rounds_min']<=full['evaluated_rounds_max']<=10000
    raw=docs['raw_execution'];assert raw['models']==125 and raw['raw_preprocessing'] is True and raw['full_training_limits'] is True and raw['strict_json'] is True
    graph_report=docs['graph_execution'];check_hashes(ROOT,graph_report['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('vsb_*.py')}<=set(graph_report['code_sha256'])
    assert graph_report['source_version_id']==SOURCE_VERSION and graph_report['source_content_hash']==SOURCE_HASH
    assert graph_report['checks']==dict(actual_runner_nodes=2,models=125,raw_preprocessing=True,full_training_limits=True,synthetic_query_rows=8,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True)
    from sciona.vsb_graph import build_vsb_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.vsb_execution as provider
    graph=build_vsb_graph();assert encode_execution_graph(graph)[0]==graph_report['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        fn=getattr(provider,'vsb_'+node.node_id)
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs] and all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_vsb_'+node.node_id)(witness)
    assert witness=={'kind':'VSB.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==graph_report['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/vsb-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert len(dependencies)==12 and dependencies==env['dependency_versions']==dependency_closure()
    assert sha(requirements)==env['requirements_sha256'] and env['verifier_sha256']==sha(ROOT/'scripts/validate_vsb_environment.py')
    assert env['python_version']==platform.python_version() and env['platform']==platform.system() and env['machine']==platform.machine()
    assert env['execution_device']=='cpu' and env['actual_root_imports_verified'] is True and env['declared_dependency_constraints_satisfied'] is True
    assert len(env['license_sha256'])==36;check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_vsb_execution.py','scripts/validate_vsb_environment.py','requirements/vsb-execution.txt']+['docs/licenses/'+p for p in env['license_sha256']]
    return dict(format='vsb-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=graph_report['serialized_graph_sha256'],provider_sha256=sha(provider_path),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=STAGE_MAPPING,
        dependencies=dependencies,limitations=LIMITATIONS,evidence_sha256={f'competition_vsb_{s}.json':sha(reviews/f'competition_vsb_{s}.json') for s in SUFFIXES},auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_vsb_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],tier=result['proposed_tier'],catalog_mutations=0)))

"""Automated source/computation review for corrected CHAMPS Tier 3 publication."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE_VERSION='af2888ee-499c-5db7-8576-f4096e1f8804'
SOURCE_HASH='d1891c5f46afe2b677bce08e67522a016bd798bb654d43f4060822f957eb30a9'
EVIDENCE=('source_pins','ensemble_execution','loss_execution','prediction_execution',
          'preprocessing_runtime_execution','training_runtime_streaming_execution',
          'lifecycle_execution','graph_execution','environment')
LIMITATIONS=[
 'Automated Tier 3 Community reconstruction. No Tier 1 human certification or Tier 2 usage qualification; original inaccurate intake remains draft.',
 'Pinned Bosch graph-transformer solution corrects the intake SchNet/LightGBM claims. All thirteen full configurations are retained; no original pretrained checkpoint or historical accuracy parity is claimed.',
 'Explicit corrected triplet packing/model indices and chunked evaluation; modern OpenBabel/pandas interfaces. Caller-seeded batching and complete structured checkpoints replace launcher global state and model pickles.',
 'Original chemistry requires supported connected motifs and fixed 29/406/54/117 capacities. Degenerate or absent motifs, unsupported chemistry, out-of-vocabulary labels and nonfinite scaling can fail. This is not a general molecular chemistry predictor.',
 'Original per-record bond corrections are excluded. Optional manual corrections in lower-level preprocessing are not exposed in this version of the pipeline payload; original-record parity is not claimed.',
 'Fitted vocabulary/scaling precede the source population split. This preserves source behavior rather than claiming leakage-free validation. The eight supervised primary type slots are explicitly reserved for smaller populations.',
 'Source inverse-square-root schedule is initialized but never advanced after warmup; this behavior is preserved. Original quadruplet cutout and chunked log-type-loss training combinations remain unsupported.',
 'Graph evidence uses one explicitly configured synthetic SGD epoch for every full model. Sixteen stochastic optimizer/scheduler resume combinations and validation selection controls were checked separately with analytic synthetic inputs, not historical training runs.',
 'Validation selection requires all eight finite type metrics. Full-population mode retains the last epoch; finite aggregate MAE alone is not represented as complete competition macro-metric coverage.',
 'Per-model prediction rounding to six decimals precedes the coupling-specific central-five ensemble, matching original CSV precision. Opaque caller IDs are remapped to exactly representable internal float32 integers.',
 'CPU execution requires locally provisioned hash-verified software and pinned dependencies in the matcher environment. No downloads, GPU/mixed-precision qualification, clean-environment installation or resource-cap guarantee is claimed.',
 'Geometry, fitted state, checkpoints, training outputs and predictions are private runtime data. Publication evidence contains synthetic fixtures and aggregate checks only.'
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    if not __debug__: raise RuntimeError('Review assertions must be enabled')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_champs_{s}.json').read_text()) for s in EVIDENCE}
    from sciona.champs_source_runtime import ChampsSourceRuntime,PINS_SHA256
    from sciona.champs_ensemble import MODEL_ORDER
    runtime=ChampsSourceRuntime(Path('/private/tmp/sciona_champs_source'))
    pins=docs['source_pins']
    assert pins['commit']==runtime.commit=='4a42e18b5b88043fb40ec15289216a1d88789698'
    assert pins['repository']=='boschresearch/BCAI_kaggle_CHAMPS' and pins['license']['spdx_id']=='MIT'
    assert len(pins['pins'])==71 and sha(reviews/'competition_champs_source_pins.json')==PINS_SHA256
    assert runtime.sources['LICENSE']==(ROOT/'docs/licenses/CHAMPS-MIT.txt').read_bytes()
    for path,marker,notice in [('src/xyz2mol.py','from rdkit import Chem','CHAMPS-xyz2mol-MIT.txt'),
                              ('src/utils/radam.py','import math','CHAMPS-RAdam-Apache-2.0.txt')]:
        text=runtime.sources[path].decode()
        assert text[:text.index(marker)]==(ROOT/'docs/licenses'/notice).read_text()
    validators={'ensemble_execution':'ensemble','loss_execution':'loss','prediction_execution':'prediction',
        'preprocessing_runtime_execution':'preprocessing_runtime','training_runtime_streaming_execution':'training_runtime',
        'lifecycle_execution':'lifecycle'}
    for suffix,name in validators.items():
        assert docs[suffix]['source_commit']==runtime.commit
        assert docs[suffix]['validator_sha256']==sha(ROOT/f'scripts/validate_champs_{name}.py')
    ensemble=docs['ensemble_execution']['cases']
    assert len(ensemble)==12 and all(x['records']==40 and x['exact_match'] is True for x in ensemble)
    loss=docs['loss_execution']['results']
    assert {x['objective'] for x in loss}=={'global_mae','mean_log_type_mae'} and len(loss)==2
    assert all(all(x[k] is True for k in ('all_eight_types','unequal_type_counts','scalar_loss_and_gradient_match','auxiliary_and_padding_excluded')) for x in loss)
    prediction=docs['prediction_execution']
    assert prediction['synthetic_predictions']==6
    assert all(prediction[k] is True for k in ('source_predictor_exact_match','partial_final_batch','reordered_ids_aligned','six_decimal_source_boundary','duplicate_missing_rows_rejected'))
    assert prediction['runtime_sha256']==sha(ROOT/'sciona/champs_prediction.py')
    preprocessing=docs['preprocessing_runtime_execution']
    assert all(preprocessing[k] is True for k in ('reusable_runtime_matches_reference','fitted_state_replay_and_inference','finite_packed_tensors','translation_checks'))
    assert preprocessing['runtime_sha256']==sha(ROOT/'sciona/champs_preprocessing.py')
    training=docs['training_runtime_streaming_execution']
    assert training['runtime_sha256']==sha(ROOT/'sciona/champs_training.py')
    assert len(training['results'])==16 and {(r['optimizer'],r['scheduler']) for r in training['results']}=={(o,s) for o in ('SGD','Adam','Adagrad','RAdam') for s in ('constant','cosine','inv_sqrt','dev_perf')}
    assert all(all(r[k] is True for k in ('exact_resume','cutout_and_dropout_enabled','caller_rng_and_inputs_preserved')) for r in training['results'])
    lifecycle=docs['lifecycle_execution']
    assert [x['variant'] for x in lifecycle['results']]==list(MODEL_ORDER) and lifecycle['ensemble_predictions']==12
    assert all(x['full_population_fit'] and x['complete_checkpoint_reloaded'] and x['restored_preprocessing_inference'] and x['predictions']==12 for x in lifecycle['results'])
    for name,digest in lifecycle['runtime_hashes'].items(): assert sha(ROOT/'sciona'/f'{name}.py')==digest
    graph=docs['graph_execution']
    assert graph['status']=='passed'
    assert graph['checks']==dict(actual_runner_nodes=2,full_model_lifecycles=13,synthetic_predictions=12,complete_checkpoint_reload=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True)
    paths={str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('champs_*.py')}
    assert paths<=set(graph['code_sha256'])
    for name,digest in graph['code_sha256'].items():
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT) and sha(path)==digest
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/champs_execution.py'
    assert sha(provider)==graph['provider_sha256']
    from sciona.champs_graph import build_champs_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_champs_graph())[0]==graph['serialized_graph_sha256']
    env=docs['environment']
    assert env['status']=='passed' and env['tests_passed']==50
    assert set(env['test_sha256'])=={str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_champs_*.py')}
    for name,digest in env['test_sha256'].items(): assert sha(ROOT/name)==digest
    dependencies=dict(line.split('==') for line in (ROOT/'requirements/champs-execution.txt').read_text().splitlines() if line and not line.startswith('#'))
    assert dependencies==env['direct_runtime_versions']
    assert all(importlib.metadata.version(n)==v for n,v in dependencies.items())
    mapping=[('spatial_graphing','XYZ bond inference, chemistry tags/charges, atom/bond/triplet/quadruplet motifs and corrected fixed packing'),
             ('target_scaling','Per-subtype mean/sample-standard-deviation scaling, not a distributional Gaussian guarantee'),
             ('message_passing_neural_network_mpnn','Thirteen full configured graph transformers, not SchNet'),
             ('attention_mechanism','Distance-biased multi-head attention across typed motifs with grouped subtype prediction heads'),
             ('ensembling','Six-decimal unscaled predictions, fixed coupling-specific model masks and central-five mean, no LightGBM')]
    auxiliary=['scripts/review_champs_execution.py','requirements/champs-execution.txt',
               'docs/licenses/CHAMPS-MIT.txt','docs/licenses/CHAMPS-xyz2mol-MIT.txt','docs/licenses/CHAMPS-RAdam-Apache-2.0.txt']
    return {'format':'champs-semantic-review.v1','review_source':'automated','proposed_tier':3,
        'verdict':'acceptable_with_limits','source_version_id':SOURCE_VERSION,'source_hash':SOURCE_HASH,
        'source_commits':{'winning_solver':runtime.commit},'serialized_graph_sha256':graph['serialized_graph_sha256'],
        'provider_sha256':sha(provider),'provider_package_sha256':sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),
        'intake_stage_mapping':[{'intake':s,'realization':r} for s,r in mapping],
        'dependencies':dependencies,'limitations':LIMITATIONS,
        'evidence_sha256':{f'competition_champs_{s}.json':sha(reviews/f'competition_champs_{s}.json') for s in EVIDENCE},
        'auxiliary_sha256':{p:sha(ROOT/p) for p in auxiliary},'catalog_mutations':0}


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_champs_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verdict':result['verdict'],'proposed_tier':3,'intake_stages':5,'catalog_mutations':0}))

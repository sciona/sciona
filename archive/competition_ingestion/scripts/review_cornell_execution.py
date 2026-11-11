"""Automated source-bound Tier3 review of the corrected complete Cornell graph."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE_VERSION='5326bafb-d26f-5202-85c9-6f903df0fc77'
SOURCE_HASH='0e75554c472b820a39e97f3807ebc13ebd7d915af068b36ca4d364997b9b1690'
COMMIT='4ad1aa4ed99bc097289c7593c55bc09234e0fc59'
SUFFIXES=['source_pins','historical_dependencies','training_inventory','source_execution','loss_execution',
    'sampling_execution','population_execution','augmentation_execution','augmentation_runtime_execution',
    'schedule_execution','validation_execution','model_runtime_execution','training_execution','voting_execution',
    'lifecycle_execution','graph_execution','environment']
LIMITATIONS=[
 'Automated Tier3 Community corrected reconstruction; no Tier1 human certification or Tier2 community-usage qualification. Original inaccurate intake remains draft.',
 'Winner uses thirteen full DenseNet121 SED models, not intake ResNeSt50 or learned per-species thresholds. Four model-only continuation phases produce seventeen total training phases.',
 'Explicit caller fourfold/fivefold populations, epoch count, even batch size and initialization are required. Synthetic evidence uses two epochs per phase and offline random initial weights; no historical training, pretrained checkpoint or competition accuracy parity.',
 'Provisioned backbone state must match complete DenseNet121; its external training provenance is not certified by shape validation. Synthetic initialization is an explicit selectable mode, never an implicit substitute for pretrained state.',
 'Inputs are private already-resampled mono32kHz arrays. No audio decoder/resampling or external pretrained/noise download is included. Class indices retain264source output slots; caller supplies valid labels and fold membership.',
 'Pinned audiomentations0.11.0 recipes and historical Ignite schedule semantics are preserved. Per-instance RNG and explicit epoch seeds replace original worker entropy; no original stochastic-run identity is claimed.',
 'Source excludes the final possible random crop offset, uses training drop-last and balanced sampling with replacement, and validates first two clips. Augmented loss primary dropout remains active in validation.',
 'Source two-epoch validation cadence and five ranked checkpoints are retained. Execution selects best score/latest retained tie, replacing hand-picked historical checkpoints. Complete mid-epoch population/worker-state resume is not claimed.',
 'Independent array voting uses0.3clip/frame thresholds and four votes, with a second whole-record vote over voted windows. Singleton/empty events are supported; half-open chunk and duration boundaries exclude padding/endpoints. These are explicit corrected behavior, not buggy notebook parity.',
 'Linked notebook reuse license remains unresolved; its code is comparison evidence and is not embedded in runtime. Training repository MIT, bundled PANNs/mixup notices and historical dependency licenses are retained.',
 'Ten source evaluation copies and NumPy mean are retained despite eval-mode augmentation being disabled. Attention probability guards allow eight float32 ULPs of rounding without changing threshold decisions.',
 'Noise banks must be nonempty, finite positive float32 RMS and at least0.1seconds; nonfinite runtime results fail. Degenerate or extreme inputs may fail; no resource-cap, GPU/mixed-precision or clean-install qualification.',
 'Private records, noise arrays, checkpoints and predictions stay runtime values or temporary files. Committed evidence contains only synthetic checks, software identities and aggregate results.'
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit():
    if not __debug__:raise RuntimeError('Assertions required for semantic review')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_cornell_{s}.json').read_text()) for s in SUFFIXES}
    from sciona.cornell_augmentation import PINS,DEPENDENCIES
    from sciona.cornell_lifecycle import INVENTORY
    assert sha(reviews/'competition_cornell_source_pins.json')==PINS
    assert sha(reviews/'competition_cornell_historical_dependencies.json')==DEPENDENCIES
    assert sha(reviews/'competition_cornell_training_inventory.json')==INVENTORY
    pins=docs['source_pins'];assert pins['repository']=='ryanwongsa/kaggle-birdsong-recognition' and pins['commit']==COMMIT and pins['license']['spdx_id']=='MIT'
    cache=Path('/private/tmp/sciona_cornell_source')
    for pin in pins['pins']:assert sha(cache/pin['software_path'])==pin['sha256']
    assert (cache/'LICENSE').read_bytes()==(ROOT/'docs/licenses/Cornell-first-place-MIT.txt').read_bytes()
    for dep in docs['historical_dependencies']:
        base=Path('/private/tmp/sciona_cornell_dependencies')/dep['wheel'].split('-')[0]
        for pin in dep['pins']:assert sha(base/pin['software_path'])==pin['sha256']
    inventory=docs['training_inventory']['members'];assert len(inventory)==13 and [m['ordinal'] for m in inventory]==list(range(13))
    assert sum(m['parameters']['loads_prior_checkpoint'] for m in inventory)==4
    validators={s:s.removesuffix('_execution') for s in SUFFIXES if s.endswith('_execution') and s!='graph_execution'}
    validators['loss_execution']='losses'
    for suffix,name in validators.items():
        d=docs[suffix];assert d['validator_sha256']==sha(ROOT/f'scripts/validate_cornell_{name}.py')
        if 'source_commit' in d:assert d['source_commit']==COMMIT
        if 'status' in d:assert d['status']=='passed'
        hashes=d.get('runtime_sha256',{})
        if isinstance(hashes,str):
            runtime={'population':'population','sampling':'sampling','augmentation_runtime':'augmentation','schedule':'schedule','validation':'validation','voting':'voting'}[name]
            assert hashes==sha(ROOT/f'sciona/cornell_{runtime}.py')
        else:
            for path,digest in hashes.items():assert sha(ROOT/path if path.startswith('sciona/') else ROOT/'sciona'/path)==digest
    s=docs['source_execution'];assert s['complete_densenet121_topology'] and s['classes']==264
    assert all(s[k] is True for k in ['stft_reference_passed','logmel_reference_passed','mixed_waveform_forward_backward','finite_backbone_attention_gradients'])
    loss=docs['loss_execution'];assert len(loss['cases'])==12 and all(c['scalar_and_gradient_match'] for c in loss['cases'])
    assert loss['both_losses_secondary_factor_respected'] and loss['augmented_loss_primary_dropout_active_in_eval']
    sampling=docs['sampling_execution'];assert len(sampling['checks'])==8 and all(c['exact_source_match'] and c['rng_advance_exact'] for c in sampling['checks'])
    population=docs['population_execution'];assert population['source_sampler_seeds']==16 and population['invalid_cases_rejected']==5 and population['exact_source_indices']
    aug=docs['augmentation_execution'];assert len(aug['recipes'])==2 and all(c['seeds']==24 and c['all_positions_activated'] and c['exact_seed_replay'] for c in aug['recipes'])
    assert aug['background_rms_repeat_reference'] and aug['short_noise_fade_rms_reference']
    aug=docs['augmentation_runtime_execution'];assert len(aug['cases'])==2 and all(c['seeds']==12 and c['exact_source_match'] and c['exact_rng_resume'] for c in aug['cases']) and aug['invalid_noise_banks_rejected']==5
    assert aug['reference_validator_sha256']==sha(ROOT/'scripts/validate_cornell_augmentation.py')
    schedule=docs['schedule_execution'];assert len(schedule['cases'])==16 and all(c['max_abs_error']<1e-15 for c in schedule['cases']) and schedule['invalid_schedules_rejected']==7
    validation=docs['validation_execution'];assert len(validation['cases'])==6 and all(c['source_match'] for c in validation['cases'])
    model=docs['model_runtime_execution'];assert model['complete_densenet121'] and model['adamw_amsgrad_resume_exact'] and model['model_only_continuation_preserves_weights_resets_optimizer']
    training=docs['training_execution'];assert len(training['phases'])==3 and all(c['full_30_second_training'] and c['complete_checkpoint_reload'] and c['optimizer_steps']==2 for c in training['phases'])
    voting=docs['voting_execution'];assert voting['models_compared']==13 and voting['window_counts_exact'] and voting['whole_record_decisions_exact']
    lifecycle=docs['lifecycle_execution'];assert (lifecycle['models_completed'],lifecycle['training_phases'],lifecycle['continuation_phases'])==(13,17,4)
    assert all(lifecycle[k] is True for k in ['all_source_configurations','full_30_second_training','ten_copy_inference','all_selected_checkpoints_reloaded'])
    graph=docs['graph_execution'];assert graph['checks']==dict(actual_runner_nodes=2,models=13,training_phases=17,continuations=4,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True)
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('cornell_*.py')}<=set(graph['code_sha256'])
    for name,digest in graph['code_sha256'].items():
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT) and sha(path)==digest
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/cornell_execution.py'
    assert sha(provider)==graph['provider_sha256']
    from sciona.cornell_graph import build_cornell_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_cornell_graph())[0]==graph['serialized_graph_sha256']
    env=docs['environment'];assert env['status']=='passed' and env['tests_passed']==31
    assert set(env['test_sha256'])=={'tests/test_cornell_contract.py','tests/test_cornell_voting.py','tests/test_cornell_training_controls.py'}
    for name,digest in env['test_sha256'].items():assert sha(ROOT/name)==digest
    dependencies=dict(line.split('==') for line in (ROOT/'requirements/cornell-execution.txt').read_text().splitlines())
    assert dependencies==env['direct_runtime_versions'] and all(importlib.metadata.version(n)==v for n,v in dependencies.items())
    mapping=[('audio_conversion','Source waveform STFT/logmel; caller mono32kHz arrays'),('background_noise_augmentation','Pinned three/eight-transform recipes and private noise banks'),('sound_event_detection_sed_model','Thirteen full DenseNet121 SED outputs from17trainingphases'),('thresholding_optimization','Fixed0.3clip/frame threshold and4votes; no learned species thresholds'),('inference_chunking','Thirty-second chunks, ten eval copies, half-open five-second window voting')]
    auxiliary=['scripts/review_cornell_execution.py','requirements/cornell-execution.txt']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'docs/licenses').glob('Cornell-*.txt'))]
    return dict(format='cornell-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,
        source_commits={'winning_solver':COMMIT},serialized_graph_sha256=graph['serialized_graph_sha256'],provider_sha256=sha(provider),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),
        intake_stage_mapping=[dict(intake=s,realization=r) for s,r in mapping],dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_cornell_{s}.json':sha(reviews/f'competition_cornell_{s}.json') for s in SUFFIXES},auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_cornell_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3,evidence=len(SUFFIXES))))

"""Real eight-model two-stage execution on synthetic inputs and pinned weights."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from sciona.aptos_ensemble import MODEL_KEYS
from scripts.aptos_graph_execution import execute_through_graph as run_reference


def main():
    torch.set_num_threads(2)
    pins_path = Path('docs/reviews/competition_aptos_pretrained_mirror_pins.json')
    pins = json.loads(pins_path.read_text())['artifacts']
    artifact_root = Path('/private/tmp/sciona_aptos_pretrained_reference')
    paths = sorted(str(p) for p in Path('sciona').glob('aptos_*.py')) + [str(pins_path), 'scripts/validate_aptos_graph_pipeline.py']
    paths += ['scripts/aptos_graph_execution.py', 'scripts/build_aptos_execution_graph.py', 'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    keys = [f'synthetic-{i}' for i in range(7)]
    rng = np.random.default_rng(193)
    images = {key: rng.integers(0, 256, (31, 47, 3), dtype=np.uint8) for key in keys}
    with TemporaryDirectory(prefix='aptos-validation-') as work:
        print('START complete serialized APTOS graph with eight models and two stages', flush=True)
        result = run_reference(base=(keys[:2], [0, 4]), average=([keys[2]], [2]),
            grouped=(keys[3:5], [1, 1]), pseudo_keys=[keys[5]], query_keys=keys[5:], images=images,
            pretrained={family: (artifact_root / (family + '.safetensors'), row['sha256']) for family, row in pins.items()},
            first_stage_epochs={key: 5 for key in MODEL_KEYS}, seeds={key: 731 + i for i, key in enumerate(MODEL_KEYS)},
            batch_size=2, learning_rate=1e-4, lower_deviation=.5, upper_deviation=.5,
            tie_policy='upper', work_root=work)
        assert not list(Path(work).iterdir())
    assert result['checkpoint_replays_exact'] and result['temporary_checkpoints_removed']
    assert len(result['fits']) == 16
    assert [(r['stage'], r['family'], r['replica'], r['epochs']) for r in result['fits']] == [
        (stage, family, replica, epochs) for stage, epochs in [(1, 5), (2, 10)] for family, replica in MODEL_KEYS]
    assert all(len(r['losses']) == r['epochs'] and np.isfinite(r['losses']).all() for r in result['fits'])
    assert [r['optimizer_steps'] for r in result['fits']] == [5] * 8 + [30] * 8
    teacher = np.mean([result['first_stage_predictions'][key][1] for key in MODEL_KEYS], axis=0)
    expected_targets = np.r_[0., 4., (2. + teacher[0]) / 2.,
        np.clip(teacher[1:3], teacher[1:3].mean() - .5, teacher[1:3].mean() + .5), teacher[3]]
    np.testing.assert_allclose(result['second_stage_targets'], expected_targets, rtol=0, atol=1e-14)
    scores = np.mean([result['second_stage_predictions'][key][1] for key in MODEL_KEYS], axis=0)
    np.testing.assert_array_equal(result['scores'], scores)
    np.testing.assert_array_equal(result['ordinals'], np.searchsorted([.7, 1.5, 2.5, 3.5], scores, side='right'))
    assert before == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    report = {'passed': True, 'approved': False, 'catalog_mutations': 0, 'synthetic_only': True,
        'model_count': 8, 'training_stage_fits': 16, 'first_stage_epochs_per_model': 5,
        'second_stage_epochs_per_model': 10, 'optimizer_updates': sum(r['optimizer_steps'] for r in result['fits']),
        'reference_augmentation_used': True, 'checkpoint_replays_exact': True,
        'temporary_checkpoints_removed': True, 'target_role_oracle_passed': True,
        'final_ensemble_and_threshold_oracle_passed': True,
        'artifact_sha256': {family: row['sha256'] for family, row in pins.items()},
        'scope': 'Full reference budgets on synthetic population at native resolutions with all eight pretrained models. First-stage five epochs, ensemble teacher, optimizer, augmentation semantics, symmetric group bounds and threshold ties are explicit reconstruction choices. No competition accuracy or historical winning execution parity claim.',
        'sha256': before, 'graph_execution': result['_graph_evidence']}
    Path('docs/reviews/competition_aptos_graph_pipeline_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS complete serialized APTOS graph with eight models and two stages', flush=True)


if __name__ == '__main__':
    main()

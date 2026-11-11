import hashlib

import pytest
import torch

from scripts.audit_bengali_cyclegan_progress import audit_history, METRICS


@pytest.fixture
def evidence(tmp_path):
    state = {'synthetic_weight': torch.ones(2, 3)}
    path = tmp_path / 'generator_epoch_1.pt'
    torch.save(state, path)
    metrics = {k: 1. for k in METRICS}
    metrics['generator/total'] = 5.
    metrics['discriminator/total'] = 2.
    history = [dict(epoch=0, steps=2, metrics=metrics, generator_file=path.name,
                    generator_sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
    return history, tmp_path, state


def test_valid_prefix_does_not_certify_completed_fit(evidence):
    result = audit_history(*evidence)
    assert result['verified_epoch_checkpoints'] == 1
    assert result['recorded_generator_updates'] == 2
    assert result['recorded_epoch_budget_complete'] is False


@pytest.mark.parametrize('corruption', ['epoch', 'steps', 'loss', 'nonfinite', 'path', 'bytes', 'tensor'])
def test_inconsistent_history_or_checkpoint_is_rejected(evidence, corruption):
    history, runtime, state = evidence
    row = history[0]
    path = runtime / row['generator_file']
    if corruption == 'epoch':
        row['epoch'] = 1
    elif corruption == 'steps':
        row['steps'] = 1
    elif corruption == 'loss':
        row['metrics']['generator/total'] = 4.
    elif corruption == 'nonfinite':
        row['metrics']['generator/font'] = float('nan')
    elif corruption == 'path':
        row['generator_file'] = '../generator_epoch_1.pt'
    elif corruption == 'bytes':
        path.write_bytes(b'synthetic changed artifact')
    elif corruption == 'tensor':
        torch.save({'synthetic_weight': torch.ones(3, 2)}, path)
        row['generator_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        audit_history(history, runtime, state)

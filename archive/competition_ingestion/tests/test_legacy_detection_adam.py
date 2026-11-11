import copy

import pytest
import torch

from sciona.legacy_detection_adam import LegacyDetectionAdam


def test_restored_optimizer_continues_exactly_and_preserves_loaded_rate():
    parameter = torch.nn.Parameter(torch.tensor([.5, -.2]))
    optimizer = LegacyDetectionAdam([parameter], lr=8e-5)
    for _ in range(7):
        parameter.grad = torch.tensor([.01, -.6])
        optimizer.step()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = LegacyDetectionAdam([restored_parameter], lr=1e-5)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert restored.param_groups[0]['lr'] == 8e-5
    for _ in range(11):
        for p in (parameter, restored_parameter):
            p.grad = torch.tensor([-.3, .8])
        optimizer.step()
        restored.step()
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
    for key in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(optimizer.state[parameter][key], restored.state[restored_parameter][key], rtol=0, atol=0)
    assert optimizer.state[parameter]['step'] == restored.state[restored_parameter]['step'] == 18


@pytest.mark.parametrize('rate', [-1, float('nan'), float('inf'), True])
def test_invalid_rate_rejected(rate):
    with pytest.raises(ValueError):
        LegacyDetectionAdam([torch.nn.Parameter(torch.zeros(1))], lr=rate)

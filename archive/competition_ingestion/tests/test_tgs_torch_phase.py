import numpy as np
import pytest
import torch

from sciona.tgs_torch_models import TGSResNet34
from sciona.tgs_torch_phase import train_phase
from sciona.tgs_metrics import torch_validation_metric


def test_metric_empty_and_epsilon_boundary():
    truth = torch.tensor([1., 0., 0., 0.]).reshape(2, 1, 1, 2)
    logits = torch.tensor([1., 1., -1., -1.]).reshape_as(truth)
    result = torch_validation_metric(logits, truth)
    np.testing.assert_array_equal(result['per_image'], [0., 1.])


@pytest.mark.parametrize('variant', [5, 3])
def test_real_model_cycles_diagnostic_budget(variant):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(374)
        rng = np.random.default_rng(847)
        images = rng.random((2, 101, 101), dtype=np.float32)
        masks = np.zeros_like(images)
        masks[:, :, 50:] = 1
        model = TGSResNet34(variant)
        result = train_phase(model, images[:1], masks[:1], images[1:], masks[1:],
                             controls=dict(epoch=2, batch_size=1, snapshot=2, max_lr=.001, min_lr=.0001,
                                           checkpoint_policy='cycle_end' if variant == 5 else 'cycle_best'), rng=rng)
        assert result['optimizer_updates'] == 2 and result['actual_epochs'] == 2
        assert list(result['cycle_states']) == [1, 2]
        assert [row['learning_rate'] for row in result['history']] == [.001, .001]
        assert all(np.isfinite(row['validation_loss']) for row in result['history'])
        key = 'stem.0.weight'
        if variant == 5:
            torch.testing.assert_close(result['cycle_states'][2][key], model.state_dict()[key], rtol=0, atol=0)
        selected = result['cycle_states'][1][key].clone()
        with torch.no_grad():
            model.stem[0].weight.add_(1)
        torch.testing.assert_close(result['cycle_states'][1][key], selected, rtol=0, atol=0)
        model.load_state_dict(result['cycle_states'][2])
    finally:
        torch.set_num_threads(old)

import numpy as np
import torch

from sciona.tgs_resnext import TGSResNeXt50
from sciona.tgs_keras_phase import train_phase


def test_real_model_phase_integration_diagnostic_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(918)
        rng = np.random.default_rng(937)
        images = rng.uniform(0, 255, (2, 101, 101, 3)).astype(np.float32)
        masks = np.zeros((2, 101, 101), np.uint8)
        masks[:, :, 50:] = 255
        model = TGSResNeXt50()
        result = train_phase(model, images[:1], masks[:1], images[1:], masks[1:],
                             controls=dict(epochs=2, batch_size=1, learning_rate=.0001,
                                           loss_function='bce_dice', callback='snapshot', n_snapshots=1),
                             rng=rng)
        assert result['actual_epochs'] == 2 and result['optimizer_updates'] == 4
        assert list(result['periodic_states']) == [2]
        assert len(result['history']) == 2
        before = result['best_state']['prediction.weight'].clone()
        with torch.no_grad():
            model.prediction.weight.add_(1)
        torch.testing.assert_close(result['best_state']['prediction.weight'], before, rtol=0, atol=0)
        model.load_state_dict(result['periodic_states'][2])
        assert all(np.isfinite(row['validation_loss']) for row in result['history'])
    finally:
        torch.set_num_threads(previous)

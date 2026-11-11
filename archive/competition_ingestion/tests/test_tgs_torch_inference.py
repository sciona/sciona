import numpy as np
import torch

from sciona.tgs_torch_inference import predict_snapshots
from sciona.tgs_torch_models import TGSResNet34


class AnalyticLogits(torch.nn.Module):
    variant = 5
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.))
    def forward(self, images):
        return images[:, :1] + self.bias


def test_tta_crop_and_probability_average_oracle():
    image = np.broadcast_to(np.linspace(0, 1, 101, dtype=np.float32), (1, 101, 101)).copy()
    model = AnalyticLogits()
    output = predict_snapshots(model, [dict(bias=torch.tensor(-1.)), dict(bias=torch.tensor(2.))], image, batch_size=1)
    expected = (torch.from_numpy(image).sub(1).sigmoid() + torch.from_numpy(image).add(2).sigmoid()).numpy() / 2
    np.testing.assert_allclose(output, expected, atol=1e-7)
    assert output.shape == (1,101,101)


def test_real_model_snapshot_replay_and_batching():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(671)
        model = TGSResNet34(5)
        state = {k:v.clone() for k,v in model.state_dict().items()}
        images = np.random.default_rng(821).random((2,101,101), dtype=np.float32)
        first = predict_snapshots(model, [state], images, batch_size=1)
        second = predict_snapshots(model, [state], images, batch_size=1)
        np.testing.assert_array_equal(first, second)
        assert np.isfinite(first).all() and first.min() >= 0 and first.max() <= 1
    finally:
        torch.set_num_threads(previous)

import numpy as np
import torch

from sciona.tgs_keras_inference import predict_checkpoints
from sciona.tgs_resnext import TGSResNeXt50


class ConstantProbability(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(.5))
    def forward(self, images):
        return self.value.expand(len(images), 1, 224, 224)


def test_checkpoint_quantization_precedes_blending():
    model = ConstantProbability()
    images = np.zeros((1,101,101,3), np.uint8)
    result = predict_checkpoints(model, [dict(value=torch.tensor(.5)), dict(value=torch.tensor(.6))], images, batch_size=1)
    assert result.shape == (2,1,101,101)
    np.testing.assert_allclose(result[0], 127/255, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result[1], 153/255, rtol=0, atol=3e-8)


def test_real_resnext_checkpoint_inference_replays():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(537)
        model = TGSResNeXt50()
        states = [{k:v.clone() for k,v in model.state_dict().items()}]
        images = np.random.default_rng(574).uniform(0,255,(1,101,101,3)).astype(np.float32)
        first = predict_checkpoints(model, states, images, batch_size=1)
        second = predict_checkpoints(model, states, images, batch_size=1)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_allclose(first*255, np.round(first*255), atol=1e-5, rtol=0)
        assert np.isfinite(first).all()
    finally:
        torch.set_num_threads(previous)

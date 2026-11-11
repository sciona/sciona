import pytest
import torch

from sciona.tgs_torch_models import TGSResNet34
from sciona.tgs_losses import pytorch_training_loss


@pytest.mark.parametrize('variant,size', [(4, 256), (3, 256), (5, 128)])
def test_full_variant_forward_backward_and_deterministic_evaluation(variant, size):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(183)
        model = TGSResNet34(variant)
        image = torch.rand(1, 3, size, size)
        target = (torch.rand(1, 1, size, size) > .6).float()
        outputs = model(image)
        if variant == 3:
            main, pixel, classification = outputs
            assert main.shape == pixel.shape == target.shape
            assert classification.shape == (1,)
            loss = pytorch_training_loss(main, target, pixel_logits=pixel,
                                         image_probabilities=classification, image_targets=torch.ones(1))
        else:
            assert outputs.shape == target.shape
            loss = pytorch_training_loss(outputs, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.stem[0].weight.grad is not None
        assert model.stem[0].weight.grad.abs().sum() > 0
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        model.eval()
        with torch.no_grad():
            first, second = model(image), model(image)
        if variant != 3:
            first, second = (first,), (second,)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)

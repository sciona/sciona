import torch

from sciona.tgs_resnext import TGSResNeXt50
from sciona.tgs_losses import keras_elu_lovasz, keras_bce_dice


def test_source_size_both_losses_and_internal_skips():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(382)
        model = TGSResNeXt50()
        image = torch.rand(1, 3, 224, 224)
        target = (torch.rand(1, 1, 224, 224) > .5).float()
        seen = []
        handles = [layer.register_forward_pre_hook(lambda module, args: seen.append(args[1].shape[1:]))
                   for layer in list(model.decoders)[:4]]
        output = model(image)
        for handle in handles:
            handle.remove()
        assert output.shape == target.shape
        assert seen == [(1024, 14, 14), (512, 28, 28), (256, 56, 56), (64, 112, 112)]
        loss = keras_elu_lovasz(output, target)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert model.stem_conv.weight.grad.abs().sum() > 0
        model.zero_grad(set_to_none=True)
        model.probabilities = True
        probability = model(image)
        keras_bce_dice(probability, target).backward()
        assert model.prediction.weight.grad.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            first, second = model(image), model(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)

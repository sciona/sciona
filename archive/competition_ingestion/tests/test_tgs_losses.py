import math
import torch
import pytest

from sciona.tgs_losses import keras_elu_lovasz, keras_bce_dice, pytorch_training_loss


def tensor(values):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1)


def test_single_pixel_elu_penalty_and_gradient():
    logits = tensor([2., -2.]).requires_grad_()
    value = keras_elu_lovasz(logits, tensor([1., 1.]))
    assert value.item() == pytest.approx((math.exp(-1) + 4) / 2)
    value.backward()
    torch.testing.assert_close(logits.grad.flatten(), tensor([-math.exp(-1) / 2, -0.5]).flatten())


def test_global_dice_and_probability_bce():
    # BCE=log(2), global Dice=(1+1)/(1+1+1)=2/3.
    value = keras_bce_dice(tensor([.5, .5]), tensor([0., 1.]))
    assert value.item() == pytest.approx((math.log(2) + 1 / 3) / 2)


def test_auxiliary_empty_image_keeps_full_batch_denominator():
    logits = tensor([0., 0.]).requires_grad_()
    pixels = tensor([0., 0.]).requires_grad_()
    value = pytorch_training_loss(logits, tensor([0., 1.]), pixel_logits=pixels,
                                  image_probabilities=torch.tensor([.5, .5]), image_targets=torch.tensor([0., 1.]))
    assert value.item() == pytest.approx(1 + math.log(2) + .5)
    value.backward()
    torch.testing.assert_close(pixels.grad.flatten(), torch.tensor([0., -.5]))


def test_all_empty_auxiliary_has_zero_gradient():
    pixels = tensor([2., 3.]).requires_grad_()
    value = pytorch_training_loss(tensor([0., 0.]), tensor([0., 0.]), pixel_logits=pixels,
                                  image_probabilities=torch.tensor([.5, .5]), image_targets=torch.tensor([0., 0.]))
    value.backward()
    assert pixels.grad is not None and not pixels.grad.any()


def test_partial_auxiliary_inputs_rejected():
    with pytest.raises(ValueError):
        pytorch_training_loss(tensor([0.]), tensor([0.]), pixel_logits=tensor([0.]))


def test_multipixel_gradient_by_central_difference():
    logits = torch.tensor([-.7, .2, 1.8, -.1], dtype=torch.float32).reshape(1, 1, 2, 2).requires_grad_()
    target = torch.tensor([0., 1., 1., 0.]).reshape_as(logits)
    value = keras_elu_lovasz(logits, target)
    value.backward()
    for i in range(4):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus.flatten()[i] += .001
        minus.flatten()[i] -= .001
        numerical = (keras_elu_lovasz(plus, target) - keras_elu_lovasz(minus, target)) / .002
        assert logits.grad.flatten()[i].item() == pytest.approx(numerical.item(), abs=1e-4)

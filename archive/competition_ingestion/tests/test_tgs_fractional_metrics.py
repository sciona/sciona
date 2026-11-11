import math
import pytest
import torch

from sciona.tgs_losses import keras_elu_lovasz, keras_bce_dice, pytorch_training_loss
from sciona.tgs_metrics import keras_validation_metric


def test_fractional_loss_arithmetic_and_gradient():
    target = torch.tensor([.25]).reshape(1, 1, 1, 1)
    prediction = torch.ones_like(target, requires_grad=True)
    loss = keras_elu_lovasz(prediction, target)
    assert loss.item() == 2.5
    loss.backward()
    assert prediction.grad.item() == .5
    assert keras_bce_dice(torch.full_like(target, .5), target).item() == pytest.approx((math.log(2) + 2 / 7) / 2)
    with pytest.raises(ValueError):
        pytorch_training_loss(prediction, target)


def test_source_union_cap_distortion_is_visible():
    truth = torch.zeros(1, 1, 224, 224)
    truth.flatten()[:20000] = 1
    predictions = torch.ones_like(truth)
    source = keras_validation_metric(predictions, truth, score_kind='probabilities', source_union_cap=True)
    uncapped = keras_validation_metric(predictions, truth, score_kind='probabilities', source_union_cap=False)
    assert source['mean'] == 1 and source['iou'].item() > 1
    assert uncapped['mean'] == 0 and uncapped['iou'].item() < .5


def test_empty_and_strict_half_iou_keep_population_axis():
    truth = torch.tensor([1., 0., 0., 0.]).reshape(2, 1, 1, 2)
    scores = torch.tensor([1., 1., .5, .5]).reshape_as(truth)
    result = keras_validation_metric(scores, truth, score_kind='probabilities', source_union_cap=False)
    torch.testing.assert_close(result['per_image'], torch.tensor([0., 1.]))
    assert result['mean'] == .5


def test_fractional_target_metric_and_logit_threshold():
    truth = torch.tensor([.75, .25]).reshape(1, 1, 1, 2)
    prediction = torch.tensor([1., 0.]).reshape_as(truth)
    result = keras_validation_metric(prediction, truth, score_kind='logits', source_union_cap=True)
    assert result['iou'].item() == pytest.approx(.6)
    assert result['mean'] == pytest.approx(.2)

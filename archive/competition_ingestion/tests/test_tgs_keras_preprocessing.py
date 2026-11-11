import numpy as np
import pytest
import torch

from sciona.tgs_keras_preprocessing import prepare_keras
from sciona.tgs_losses import keras_bce_dice, keras_elu_lovasz


def test_bgr_scale_identity_and_reflection_padding():
    image = np.zeros((1, 101, 101, 3), np.float32)
    image[..., 0] = np.arange(101, dtype=np.float32)[None, None, :]
    image[..., 1] = 150
    image[..., 2] = 250
    result = prepare_keras(image)['images']
    assert result.shape == (1, 3, 224, 224)
    assert (result[:, 1] == 150).all() and (result[:, 2] == 250).all()
    # Reflect101 excludes the edge pixel: first outside sample equals second inside.
    torch.testing.assert_close(result[..., 15], result[..., 17])
    torch.testing.assert_close(result[..., 208], result[..., 206])
    assert result[0, 0, 16, 16] == 0 and result[0, 0, 16, 207] == 100


def test_quantized_fractional_masks_reach_both_losses():
    image = np.zeros((1, 101, 101, 3), np.uint8)
    masks = np.zeros((1, 101, 101), np.uint8)
    masks[:, :, 50:] = 255
    target = prepare_keras(image, masks)['masks']
    assert ((target > 0) & (target < 1)).any()
    torch.testing.assert_close(target * 255, (target * 255).round(), atol=1e-5, rtol=0)
    scores = torch.zeros_like(target, requires_grad=True)
    loss = keras_bce_dice(scores.sigmoid(), target) + keras_elu_lovasz(scores, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()


def test_mask_dtype_and_input_scale_are_explicit():
    image = np.zeros((1, 101, 101, 3), np.float32)
    with pytest.raises(ValueError, match='uint8'):
        prepare_keras(image, np.zeros((1, 101, 101), np.float32))
    with pytest.raises(ValueError, match='255'):
        prepare_keras(image + 256)

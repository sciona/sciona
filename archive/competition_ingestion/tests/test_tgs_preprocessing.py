import numpy as np
import torch

from sciona.tgs_preprocessing import prepare_torch


def test_position_channels_and_asymmetric_edge_padding():
    image = np.broadcast_to(np.linspace(0, 1, 101, dtype=np.float32), (1, 101, 101)).copy()
    result = prepare_torch(image, variant=5)
    x = result['images']
    assert x.shape == (1, 3, 128, 128)
    torch.testing.assert_close(x[0, 0, 13:114, 13:114], torch.from_numpy(image[0]))
    assert (x[0, 0, :, :13] == 0).all() and (x[0, 0, :, 114:] == 1).all()
    assert (x[0, 1, 0] == 0).all() and (x[0, 1, -1] == 1).all()
    torch.testing.assert_close(x[:, 2], x[:, 0] * x[:, 1])
    flipped = prepare_torch(image, variant=5, flip=True)['images']
    # Source flips before asymmetric padding, so flipping the padded tensor is wrong.
    torch.testing.assert_close(flipped[:, :, 13:114, 13:114], x[:, :, 13:114, 13:114].flip(-1))
    assert (flipped[0, 0, :, :13] == 1).all() and (flipped[0, 0, :, 114:] == 0).all()
    assert not torch.equal(flipped, x.flip(-1))


def test_training_validation_masks_and_empty_label_polarity():
    images = np.zeros((2, 101, 101), dtype=np.float32)
    masks = images.copy()
    masks[1, :, 50:] = 1
    train = prepare_torch(images, masks, variant=3, training=True)
    validation = prepare_torch(images, masks, variant=3)
    assert train['masks'].shape == (2, 1, 256, 256)
    assert validation['masks'].shape == (2, 1, 202, 202)
    torch.testing.assert_close(train['masks'][:, :, 27:229, 27:229], validation['masks'])
    torch.testing.assert_close(train['empty_image_targets'], torch.tensor([1., 0.]))
    assert ((train['masks'] == 0) | (train['masks'] == 1)).all()

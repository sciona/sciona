import numpy as np
import pytest

from sciona.tgs_augmentation import augment_torch, transform_pair


def pair():
    mask = np.zeros((101, 101), np.float32)
    mask[30:70, 40:60] = 1
    return mask.copy(), mask


@pytest.mark.parametrize('geometry,parameter', [('crop',(0,101,0,101)), ('shear',0.), ('rotate',0.)])
def test_identity_geometry(geometry, parameter):
    image, mask = pair()
    actual, target = transform_pair(image, mask, geometry=geometry, parameter=parameter)
    np.testing.assert_array_equal(actual, image)
    np.testing.assert_array_equal(target, mask)


def test_flip_and_brightness_have_distinct_mask_effects():
    image, mask = pair()
    actual, target = transform_pair(image, mask, flip=True, brightness='shift', amount=.1)
    np.testing.assert_array_equal(target, mask[:, ::-1])
    np.testing.assert_allclose(actual, np.clip(image[:, ::-1]+.1, 0, 1))
    np.testing.assert_array_equal(image, mask)


def test_seed_replay_and_nontrivial_geometry_preserve_binary_masks():
    image, mask = pair()
    first, second = np.random.default_rng(491), np.random.default_rng(491)
    changed = False
    for _ in range(30):
        a, b = augment_torch(image, mask, first)
        c, d = augment_torch(image, mask, second)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        assert np.isin(b, [0, 1]).all() and np.isfinite(a).all()
        assert a.min() >= 0 and a.max() <= 1
        changed |= not np.array_equal(b, mask)
    assert changed

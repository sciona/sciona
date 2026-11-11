import numpy as np
from sciona.tgs_keras_augmentation import photometric, augment_keras


def test_historical_float_clipping_is_explicit():
    image = np.full((101, 101, 3), 150, np.float32)
    np.testing.assert_array_equal(photometric(image), image)
    assert np.all(photometric(image, brightness=1.) == 1.)
    assert np.all(photometric(image, contrast=1.) == 1.)


def test_contrast_mean_term_includes_source_factor_three():
    image = np.full((101, 101, 3), .2, np.float32)
    np.testing.assert_allclose(photometric(image, contrast=.8), .28, atol=1e-6)


def test_seed_replay_preserves_mask_values_and_inputs():
    image = np.arange(101*101*3, dtype=np.float32).reshape(101,101,3) % 256
    original = image.copy()
    mask = np.zeros((101,101), np.uint8)
    mask[:,50:] = 255
    for seed in range(12):
        a,b = augment_keras(image,mask,np.random.default_rng(seed))
        c,d = augment_keras(image,mask,np.random.default_rng(seed))
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)
        assert np.isin(b,[0,255]).all()
    np.testing.assert_array_equal(image,original)

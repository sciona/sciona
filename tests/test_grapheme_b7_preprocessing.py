"""Synthetic pixel and RNG checks for the explicit classifier input adapter."""
import numpy as np
from PIL import Image
import pytest
import torch

from sciona.grapheme_b7_preprocessing import prepare_classifier_images
from sciona.svhn_autoaugment import SVHNAutoAugment


def fixture():
    return np.random.default_rng(59).integers(0, 256, (3, 137, 236), dtype=np.uint8)


@pytest.mark.parametrize('augment', [False, True])
def test_exact_explicit_pixel_pipeline_and_rng_continuation(augment):
    torch.set_num_threads(1)
    images = fixture()
    original = images.copy()
    actual_policy, reference_policy = SVHNAutoAugment(seed=27), SVHNAutoAugment(seed=27)
    mean, std = [.1, .3, .7], [.2, .4, .6]
    for _ in range(2):
        result = prepare_classifier_images(images, mean=mean, std=std,
                                           augmentation=actual_policy if augment else None)
        expected = []
        for source in images:
            rgb = np.repeat(source[..., None], 3, axis=2)
            if augment:
                rgb = np.asarray(reference_policy(Image.fromarray(rgb)))
            value = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()/255.
            for channel in range(3):
                value[channel] = (value[channel] - mean[channel]) / std[channel]
            expected.append(value)
        torch.testing.assert_close(result, torch.stack(expected), rtol=0, atol=0)
        assert result.shape == (3, 3, 137, 236)
        assert actual_policy.rng.getstate() == reference_policy.rng.getstate()
    np.testing.assert_array_equal(images, original)


@pytest.mark.parametrize('fault', ['geometry', 'dtype', 'empty', 'mean', 'std', 'zero', 'overflow', 'augmentation'])
def test_reject_invalid_inputs_before_rng_advances(fault):
    images = fixture()
    policy = SVHNAutoAugment(seed=37)
    before = policy.rng.getstate()
    options = dict(mean=[.5]*3, std=[.5]*3, augmentation=policy)
    if fault == 'geometry': images = images[:, :100]
    elif fault == 'dtype': images = images.astype(float)
    elif fault == 'empty': images = images[:0]
    elif fault == 'mean': options['mean'] = [float('nan')]*3
    elif fault == 'std': options['std'] = [1., 2.]
    elif fault == 'zero': options['std'] = [0.]*3
    elif fault == 'overflow': options.update(mean=[100.]*3, std=[1e-44]*3)
    else: options['augmentation'] = lambda image: image
    with pytest.raises(ValueError):
        prepare_classifier_images(images, **options)
    assert policy.rng.getstate() == before

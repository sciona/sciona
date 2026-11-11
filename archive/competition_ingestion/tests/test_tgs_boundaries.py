import numpy as np
import pytest

from sciona.tgs_boundaries import boundary_features


def test_population_and_image_sample_standardization():
    images = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    images = np.sin(images / 7) + np.cos(images / 13)
    features = boundary_features(images)
    for edge in features.values():
        np.testing.assert_allclose(edge.mean(axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(edge.std(axis=1, ddof=1), 1, atol=1e-14)
    # Swapping spatial axes swaps the paired source directions exactly.
    swapped = boundary_features(images.transpose(0, 2, 1))
    np.testing.assert_allclose(features['u'], swapped['l'])
    np.testing.assert_allclose(features['d'], swapped['r'])


def test_constant_population_is_zero_after_both_scales():
    result = boundary_features(np.ones((3, 5, 6)))
    assert all(not value.any() for value in result.values())


@pytest.mark.parametrize('bad', [np.ones((1, 5, 6)), np.ones((3, 1, 6)), np.full((3, 5, 6), np.nan)])
def test_invalid_population_rejected(bad):
    with pytest.raises(ValueError):
        boundary_features(bad)

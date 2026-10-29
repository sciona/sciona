import numpy as np
import pytest
from sciona.atoms.numerical.spatial.resampling import resample_regular_volume, witness_resample_regular_volume
from sciona.ghost.abstract import AbstractArray


def test_rounding_effective_spacing_and_witness():
    volume = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    saved = volume.copy()
    result, effective = resample_regular_volume(volume, [1.5, 1., 2.], [1., 1., 1.])
    assert result.shape == (4, 4, 10)
    np.testing.assert_array_equal(effective, [1.125, 1., 1.])
    np.testing.assert_array_equal(volume, saved)
    abstract, abstract_spacing = witness_resample_regular_volume(AbstractArray(shape=volume.shape, dtype='float64'), [1.5, 1., 2.], [1., 1., 1.])
    assert abstract.shape == result.shape and abstract.dtype == 'float64'
    np.testing.assert_array_equal(abstract_spacing, effective)


@pytest.mark.parametrize('order', range(6))
def test_constant_volume_and_dtype(order):
    volume = np.full((3, 4, 5), 17, dtype=np.uint8)
    result, _ = resample_regular_volume(volume, [1., 2., 1.], [1., 1., 1.], order)
    assert result.dtype == volume.dtype
    np.testing.assert_array_equal(result, np.full((3, 8, 5), 17, dtype=np.uint8))


@pytest.mark.parametrize('spacing,requested,order', [([0,1,1],[1,1,1],1), ([1,1,1],[100,1,1],1), ([1,1,1],[1,1,1],True), ([1,1,1],[1,1,1],6), ([1,1],[1,1,1],1)])
def test_invalid_geometry(spacing, requested, order):
    with pytest.raises(ValueError):
        resample_regular_volume(np.ones((2,3,4)), spacing, requested, order)


@pytest.mark.parametrize('volume', [np.ones((2,3)), np.empty((0,3,4)), np.full((2,3,4), np.nan), np.ones((2,3,4), dtype=complex)])
def test_invalid_volume(volume):
    with pytest.raises(ValueError):
        resample_regular_volume(volume, [1,1,1], [1,1,1])

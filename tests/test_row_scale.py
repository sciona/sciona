"""Synthetic domain-independent checks for uncentered row scaling."""
import numpy as np
import pytest
from sciona.atoms.numerical.statistics.row_scale import normalize_rows_by_scale


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('statistic', ['mean_absolute', 'population_std'])
def test_exact_ordered_numpy_reduction_and_input_preservation(dtype, order, statistic):
    x = np.array([[1, 3, -2, 4], [-7, -2, 3, 9]], dtype=dtype, order=order)
    original = x.copy()
    scale = (np.mean(np.abs(x), axis=1, keepdims=True) if statistic=='mean_absolute'
             else np.std(x, axis=1, ddof=0, keepdims=True))
    actual = normalize_rows_by_scale(x, statistic)
    np.testing.assert_array_equal(actual, x/scale)
    assert actual.dtype == dtype and not np.shares_memory(actual, x)
    np.testing.assert_array_equal(x, original)


def test_population_scale_does_not_center_values():
    np.testing.assert_array_equal(normalize_rows_by_scale(np.array([[1.,3.],[-3.,-1.]]), 'population_std'),
                                  [[1.,3.],[-3.,-1.]])


def test_mean_absolute_accepts_nonzero_constant_rows():
    np.testing.assert_array_equal(normalize_rows_by_scale(np.array([[4.,4.],[-3.,-3.]]), 'mean_absolute'),
                                  [[1.,1.],[-1.,-1.]])


@pytest.mark.parametrize('values,statistic', [
    ([[1.,2.]], 'mean_absolute'),
    (np.ones(3), 'mean_absolute'),
    (np.ones((1,2,3)), 'mean_absolute'),
    (np.empty((0,2)), 'mean_absolute'),
    (np.ones((2,0)), 'mean_absolute'),
    (np.ones((2,3), dtype=np.int32), 'mean_absolute'),
    (np.ones((2,3), dtype=np.float16), 'mean_absolute'),
    (np.array([[1., np.nan]]), 'mean_absolute'),
    (np.array([[1., np.inf]]), 'population_std'),
    (np.zeros((2,3)), 'mean_absolute'),
    (np.ones((2,3)), 'population_std'),
    (np.ones((2,3)), 'standardize'),
    (np.ones((2,3)), None),
    (np.array([[3e38, 3e38]], dtype=np.float32), 'mean_absolute'),
    (np.array([[-3e38, 3e38]], dtype=np.float32), 'population_std'),
])
def test_invalid_or_undefined_scaling_rejected(values, statistic):
    with pytest.raises(ValueError):
        normalize_rows_by_scale(values, statistic)

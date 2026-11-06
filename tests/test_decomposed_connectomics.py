"""Synthetic checks for the prefiltered source recipe boundary."""
import numpy as np
import pytest

from sciona.connectomics_runtime import prepare
from sciona.decomposed_connectomics_runtime import checked_filter, filter_configuration


def signals():
    return np.arange(30, dtype=np.float32).reshape(10, 3) / 30


@pytest.mark.parametrize('corrupt', [
    lambda x: x.astype(np.float64), lambda x: x[:-1],
    lambda x: np.full_like(x, np.nan), lambda x: x.tolist(),
])
def test_prefilter_contract_rejects_bad_arrays(corrupt):
    x = signals()
    with pytest.raises(ValueError, match='prefiltered matrix'):
        checked_filter(x, corrupt(x))


def test_prefilter_binding_is_isolated_and_rejects_other_signals():
    x = signals()
    filtered = x * 3
    expected = filtered.copy()
    callback = checked_filter(x, filtered)
    filtered[:] = 0
    result = callback(x)
    np.testing.assert_array_equal(result, expected)
    result[:] = -1
    np.testing.assert_array_equal(callback(x), expected)
    with pytest.raises(ValueError, match='different source signals'):
        callback(x + 1)


def test_recipe_configuration_preserves_input_and_explicit_taps():
    x = signals()
    original = x.copy()
    prepared = prepare(dict(version=1, signals=x, mode='tuned', directivity=False))
    values, axis, *controls = filter_configuration(prepared)
    assert axis == 0
    assert controls == [[0, -1, 1], [1., 1., 1.], [0, 1, 2, 3], [1., 1., .8, .4],
                        [0, -1, -2, 1], [1., 1., 1., 1.], [0, -1, -2, -3], [1., 1., 1., 1.]]
    assert values.dtype == np.float32 and not values.flags.writeable
    assert not np.shares_memory(values, prepared.signals)
    np.testing.assert_array_equal(x, original)

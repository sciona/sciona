import numpy as np
import pytest
from sciona.atoms.ml.calibration.prediction_features import (round_to_int32,
    widen_prediction_values, underestimation_labels, append_prediction_feature)


def test_rounding_ties_even_and_signed_values():
    x = np.array([-2.5, -1.5, -.5, .5, 1.5, 2.5], dtype=np.float32)
    np.testing.assert_array_equal(round_to_int32(x), [-2, -2, 0, 0, 2, 2])
    assert round_to_int32(x).dtype == np.int32


def test_rounding_int32_limits_without_float32_bound_coercion():
    lower = np.float32(-2**31)
    upper = np.nextafter(np.float32(2**31), np.float32(0))
    np.testing.assert_array_equal(round_to_int32(np.array([lower, upper])), [-2**31, int(upper)])
    for bad in [np.float32(2**31), np.nextafter(lower, np.float32(-np.inf)), np.float32(np.nan)]:
        with pytest.raises(ValueError): round_to_int32(np.array([bad]))


def test_training_labels_and_inference_feature_precision():
    observed = np.array([1., 2., 3.])
    rounded = np.array([0, 2, 4], dtype=np.int32)
    np.testing.assert_array_equal(underestimation_labels(observed, rounded), [1, 0, 0])
    features = np.array([[1., np.nan], [2., 3.], [4., 5.]])
    fractional = np.array([.125, 1.25, 2.75], dtype=np.float32)
    result = append_prediction_feature(features, fractional)
    np.testing.assert_array_equal(result[:, -1], fractional.astype(np.float64))
    np.testing.assert_array_equal(result[:, :2], features)
    np.testing.assert_array_equal(append_prediction_feature(features, rounded)[:, -1], rounded)


@pytest.mark.parametrize('dtype', [np.float32, np.int32])
def test_widening_exact_and_independent(dtype):
    values = np.array([0, 1, 123456], dtype=dtype)
    result = widen_prediction_values(values)
    np.testing.assert_array_equal(result, values)
    assert result.dtype == np.float64
    result[0] = 9
    assert values[0] == 0


def test_invalid_alignment_and_dtypes():
    with pytest.raises(ValueError): round_to_int32(np.array([1.], dtype=np.float64))
    with pytest.raises(ValueError): widen_prediction_values(np.array([1.], dtype=np.float64))
    with pytest.raises(ValueError): underestimation_labels(np.array([1.]), np.array([1, 2], dtype=np.int32))
    with pytest.raises(ValueError): append_prediction_feature(np.ones((2, 2)), np.ones(1, dtype=np.float32))
    with pytest.raises(ValueError): append_prediction_feature(np.array([[np.inf]]), np.ones(1, dtype=np.float32))

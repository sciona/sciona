import numpy as np
import pytest

from sciona.atoms.ml.calibration.conditional_residuals import (
    estimate_conditional_offsets, apply_conditional_offsets,
    witness_apply_conditional_offsets,
)
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.dimensions import DimensionalSignature


@pytest.mark.parametrize('baseline,scale', [(2., .01), (500., 20.)])
def test_manufacturing_dimension_and_energy_forecast_examples(baseline, scale):
    # Constructed calibration and evaluation populations; no real measurements.
    prediction = baseline + scale*np.array([1., 2., 3., 4., 5.])
    observed = prediction + scale*np.array([2., 4., -1., -3., 999.])
    probabilities = np.array([.8, .9, .1, .2, .5])
    offsets = estimate_conditional_offsets(observed, prediction, probabilities, .5)
    np.testing.assert_allclose(offsets, scale*np.array([3., 2.]), atol=1e-13)
    evaluation = baseline + scale*np.array([8., 9., 10.])
    actual = apply_conditional_offsets(evaluation, np.array([.7, .3, .5]), offsets, .5)
    np.testing.assert_allclose(actual, evaluation + scale*np.array([3., -2., 0.]), atol=1e-13)
    np.testing.assert_array_equal(evaluation, baseline + scale*np.array([8., 9., 10.]))


def test_classifier_direction_can_be_wrong_and_offsets_remain_signed():
    offsets = estimate_conditional_offsets(np.array([8., 12.]), np.array([10., 10.]), np.array([.9, .1]), .5)
    np.testing.assert_array_equal(offsets, [-2., -2.])
    np.testing.assert_array_equal(apply_conditional_offsets(np.array([20., 20.]), np.array([.9, .1]), offsets, .5), [18., 22.])


@pytest.mark.parametrize('probabilities', [np.array([.5, .5]), np.array([.8, .9]), np.array([.1, .2])])
def test_undefined_conditional_median_rejected(probabilities):
    with pytest.raises(ValueError, match='both conditional'):
        estimate_conditional_offsets(np.ones(2), np.zeros(2), probabilities, .5)


@pytest.mark.parametrize('probabilities', [np.array([-1., .2]), np.array([np.nan, .2]), np.array([.2])])
def test_invalid_probabilities_and_alignment_rejected(probabilities):
    with pytest.raises(ValueError):
        apply_conditional_offsets(np.ones(2), probabilities, np.ones(2), .5)


def test_no_hidden_output_quantization_and_overflow_rejected():
    result = apply_conditional_offsets(np.array([1.25]), np.array([.9]), np.array([.125, .2]), .5)
    np.testing.assert_array_equal(result, [1.375])
    with pytest.raises(ValueError, match='finite float64'):
        apply_conditional_offsets(np.array([1e308]), np.array([.9]), np.array([1e308, .2]), .5)


def test_witness_rejects_cross_unit_corrections():
    time = DimensionalSignature(T=1)
    length = DimensionalSignature(L=1)
    with pytest.raises(ValueError, match='compatible units'):
        witness_apply_conditional_offsets(AbstractArray(shape=(3,), dim=time), AbstractArray(shape=(3,)),
                                          AbstractArray(shape=(2,), dim=length), AbstractScalar(dtype='float64'))


def test_witness_rejects_known_empty_population():
    with pytest.raises(ValueError, match='aligned vector'):
        witness_apply_conditional_offsets(AbstractArray(shape=(0,)), AbstractArray(shape=(0,)),
                                          AbstractArray(shape=(2,)), AbstractScalar(dtype='float64'))


@pytest.mark.parametrize('threshold', [True, float('nan'), -.1, 1.1])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match='threshold'):
        apply_conditional_offsets(np.ones(2), np.array([.2, .8]), np.ones(2), threshold)


def test_calibration_and_application_populations_can_differ_in_size():
    offsets = estimate_conditional_offsets(np.array([3., -2.]), np.zeros(2), np.array([.9, .1]), .5)
    result = apply_conditional_offsets(np.zeros(3), np.array([.9, .1, .5]), offsets, .5)
    np.testing.assert_array_equal(result, [3., -2., 0.])

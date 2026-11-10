import numpy as np
import pytest

from sciona.quantile_target_encoding import (
    quantile_boundaries, quantile_midpoints, encode_quantile_targets,
    decode_quantile_probabilities,
)


def test_independent_two_class_soft_formula():
    edges = np.array([0., .5, 1.])
    targets = np.array([0., .25, .5, .75, 1.])
    actual = encode_quantile_targets(targets, edges, precision=8.)
    # With centers .25 and .75, the class log-odds are 4*(target-.5).
    right = 1/(1+np.exp(-4*(targets-.5)))
    np.testing.assert_allclose(actual, np.column_stack([1-right, right]), rtol=1e-14)


def test_training_quantiles_and_repeated_class_boundaries():
    np.testing.assert_array_equal(quantile_boundaries([0, 1, 2, 3, 4], 4), [0, 1, 2, 3, 4])
    repeated = quantile_boundaries([2, 2, 2], 3)
    np.testing.assert_array_equal(repeated, [2, 2, 2, 2])
    np.testing.assert_allclose(encode_quantile_targets([2], repeated), [[1/3]*3])


def test_hard_boundary_asymmetry_is_preserved():
    actual = encode_quantile_targets([-1, 0, .5, 1, 2], [0, .5, 1], soft=False)
    np.testing.assert_array_equal(actual, [[0, 0], [0, 0], [1, 0], [0, 1], [0, 0]])
    assert actual.dtype == np.float32


def test_transfer_requires_precision_unit_conversion():
    # Constructed normalized demand versus temperature: no application records.
    demand = np.array([.1, .4, .8])
    demand_edges = [0., .5, 1.]
    encoded = encode_quantile_targets(demand, demand_edges, precision=8.)
    temperature = 10*demand-20
    temperature_edges = 10*np.array(demand_edges)-20
    transferred = encode_quantile_targets(temperature, temperature_edges, precision=8/100)
    np.testing.assert_allclose(encoded, transferred, atol=1e-14)
    np.testing.assert_allclose(decode_quantile_probabilities(transferred, temperature_edges),
        10*decode_quantile_probabilities(encoded, demand_edges)-20)


def test_decode_independent_weighted_expectation():
    np.testing.assert_array_equal(quantile_midpoints([0, 2, 6]), [1, 4])
    np.testing.assert_allclose(decode_quantile_probabilities([[.25, .75]], [0, 2, 6]), [3.25])


@pytest.mark.parametrize('values', [[np.nan], [np.inf], [], ['1'], [[1]]])
def test_reject_invalid_targets(values):
    with pytest.raises(ValueError):
        quantile_boundaries(values, 2)


def test_reject_underflow_and_invalid_probability_contract():
    with pytest.raises(ValueError, match='underflow'):
        encode_quantile_targets([1e6], [0, .5, 1])
    for probabilities in ([[.2, .2]], [[-.1, 1.1]], [[np.nan, .5]], [[1.]]):
        with pytest.raises(ValueError):
            decode_quantile_probabilities(probabilities, [0, .5, 1])

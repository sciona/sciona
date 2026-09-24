import numpy as np
import pytest
from sciona.atoms.ml.model_selection.masked_operations import (
    intersect_masks, fit_masked_regression, fit_masked_binary, estimate_masked_offsets)
from sciona.atoms.ml.xgboost.model_io import (
    fit_regression_model, fit_binary_model, predict_regression_model, predict_binary_model)
from sciona.atoms.ml.calibration.conditional_residuals import estimate_conditional_offsets


@pytest.mark.parametrize('kind', ['regression', 'binary'])
def test_masked_fit_matches_explicit_population_and_excludes_other_rows(kind):
    rng = np.random.default_rng(211)
    x = rng.normal(size=(80, 3))
    y = x[:, 0]-2*x[:, 1]
    if kind == 'binary': y = (y > 0).astype(np.int64)
    selected = np.arange(80)%3 != 0
    names = ['a', 'b', 'c']
    masked, fit, predict = ((fit_masked_regression, fit_regression_model, predict_regression_model)
        if kind == 'regression' else (fit_masked_binary, fit_binary_model, predict_binary_model))
    expected = fit(x[selected], y[selected], names)
    actual = masked(x, y, names, selected)
    np.testing.assert_array_equal(predict(x[:8], names, actual), predict(x[:8], names, expected))
    changed = x.copy(); changed[~selected] = 1e6
    other = masked(changed, y, names, selected)
    np.testing.assert_array_equal(predict(x[:8], names, other), predict(x[:8], names, actual))


def test_masked_calibration_ignores_unselected_extreme_residuals():
    observed = np.array([10., 20., 1e20, -1e20])
    predicted = np.array([8., 25., 0., 0.])
    probability = np.array([.8, .2, .9, .1])
    selection = np.array([True, True, False, False])
    np.testing.assert_array_equal(estimate_masked_offsets(observed, predicted, probability, .5, selection), [2., 5.])
    np.testing.assert_array_equal(estimate_masked_offsets(observed, predicted, probability, .5, selection),
        estimate_conditional_offsets(observed[selection], predicted[selection], probability[selection], .5))


def test_mask_intersection_allows_empty_without_casting_integers():
    np.testing.assert_array_equal(intersect_masks(np.array([True, False]), np.array([False, True])), [False, False])
    with pytest.raises(ValueError): intersect_masks(np.array([1, 0]), np.array([True, False]))


def test_invalid_or_insufficient_selections_rejected():
    x, y = np.ones((3, 2)), np.arange(3, dtype=np.float64)
    for mask in [np.array([False]*3), np.array([True]*2), np.array([1]*3)]:
        with pytest.raises(ValueError): fit_masked_regression(x, y, ['a', 'b'], mask)
    with pytest.raises(ValueError):
        fit_masked_binary(x, np.array([0, 1, 0], dtype=np.int64), ['a', 'b'], np.array([True, False, True]))
    with pytest.raises(ValueError):
        estimate_masked_offsets(y, y, np.array([.9, .1, .9]), .5, np.array([True, False, True]))

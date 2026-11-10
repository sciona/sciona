import pickle

import numpy as np
import pytest

from sciona.positive_log_blend import PositiveLogBlend, fit_positive_log_blend


def test_hand_calculated_affine_log_oracle_and_clipping():
    blend = PositiveLogBlend((1., 2.), np.log(2.), (0., 10.))
    actual = blend.predict([[1., 2.], [-1., 20.]])
    np.testing.assert_allclose(actual, [2*2*3**2-1, 2*1*11**2-1])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(blend)).predict([[1., 2.]]), actual[:1])


def test_fit_positive_coefficients_and_explicit_empty_query():
    rng = np.random.default_rng(877)
    scores = rng.uniform(1, 10, (80, 3))
    targets = np.expm1(.3 + np.log1p(scores) @ [0.6, 0.2, 0.1])
    blend = fit_positive_log_blend(scores, targets, clip_bounds=(0, 20))
    assert all(c >= 0 for c in blend.coefficients)
    assert blend.predict(np.empty((0, 3))).shape == (0,)
    assert np.all(blend.predict(scores+1) >= blend.predict(scores))


def test_constant_target_keeps_intercept():
    blend = fit_positive_log_blend([[1, 2], [2, 3], [3, 4]], [7, 7, 7], clip_bounds=(0, 10))
    np.testing.assert_allclose(blend.predict([[9, 9]]), [7])


def test_cross_domain_response_coordinate_shift():
    rng = np.random.default_rng(881)
    scores = rng.uniform(1, 5, (100, 2))
    targets = np.expm1(.4 + np.log1p(scores) @ [.7, .2])
    first = fit_positive_log_blend(scores, targets, clip_bounds=(0, 10), tolerance=1e-10, max_iter=10000)
    second = fit_positive_log_blend(3*(1+scores)-1, 3*(1+targets)-1,
                                  clip_bounds=(2, 32), tolerance=1e-10, max_iter=10000)
    np.testing.assert_allclose(second.predict(3*(1+scores)-1), 3*(1+first.predict(scores))-1, rtol=1e-8)


@pytest.mark.parametrize('bounds', [(0, 0), (-1, 2), (2, 1), (0, np.inf)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        fit_positive_log_blend([[1], [2]], [1, 2], clip_bounds=bounds)


def test_bad_shapes_nonfinite_and_overflow():
    blend = PositiveLogBlend((1.,), 0., (0., 10.))
    for value in ([[1, 2]], [[np.nan]], [[np.inf]], [1, 2]):
        with pytest.raises(ValueError):
            blend.predict(value)
    with pytest.raises(ValueError):
        PositiveLogBlend((1.,), 1000., (0., 10.)).predict([[1.]])
    with pytest.raises(ValueError):
        fit_positive_log_blend([[1], [2]], [-1, 2], clip_bounds=(0, 10))

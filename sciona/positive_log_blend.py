"""Positive-coefficient Lasso blending in log1p response coordinates.

Adapted from mercari/main_helpers.py in pjankiewicz/mercari-solution,
commit 374301ad1c32cbc93dcc40313d5d7bb9c5503746.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

Prediction columns must have stable identities and shared response units.
Callers supply held-out or out-of-fold calibration predictions; this numeric
core cannot establish population independence. Clip bounds are explicit policy
in response units. Nonnegative coefficients need not sum to one, and the free
intercept means the result is not a convex average or guaranteed nonnegative.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


def _bounds(bounds):
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or not 0 <= values[0] < values[1]:
        raise ValueError('Finite ordered nonnegative clipping bounds required')
    return tuple(float(v) for v in values)


def _scores(scores, bounds):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Finite two-dimensional prediction matrix required')
    return np.log1p(np.clip(values, *bounds))


@dataclass(frozen=True)
class PositiveLogBlend:
    coefficients: tuple
    intercept: float
    clip_bounds: tuple

    def __post_init__(self):
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all() or (coefficients < 0).any():
            raise ValueError('Finite nonnegative coefficient vector required')
        if not np.isscalar(self.intercept) or not np.isfinite(self.intercept):
            raise ValueError('Finite scalar intercept required')
        object.__setattr__(self, 'coefficients', tuple(float(v) for v in coefficients))
        object.__setattr__(self, 'intercept', float(self.intercept))
        object.__setattr__(self, 'clip_bounds', _bounds(self.clip_bounds))

    def predict(self, predictions):
        scores = _scores(predictions, self.clip_bounds)
        if scores.shape[1] != len(self.coefficients):
            raise ValueError('Prediction width differs from fitted blend')
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.expm1(scores @ np.asarray(self.coefficients) + self.intercept)
        if not np.isfinite(result).all():
            raise ValueError('Inverse response transform overflowed')
        return result


def fit_positive_log_blend(predictions, targets, *, clip_bounds, alpha=0.0001,
                           max_iter=1000, tolerance=0.0001, seed=9999):
    """Fit a frozen blend on caller-designated calibration scores and targets."""
    bounds = _bounds(clip_bounds)
    scores = _scores(predictions, bounds)
    y = np.asarray(targets, dtype=np.float64)
    if y.shape != (len(scores),) or len(y) < 2 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('At least two aligned finite nonnegative targets required')
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive finite regularization and tolerance required')
    if type(max_iter) is not int or max_iter < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid solver iteration count or random seed')
    estimator = Lasso(alpha=alpha, precompute=True, max_iter=max_iter, positive=True,
                      random_state=seed, selection='random', fit_intercept=True, tol=tolerance)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(scores, np.log1p(y))
    return PositiveLogBlend(tuple(estimator.coef_), float(estimator.intercept_), bounds)

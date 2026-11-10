"""Quantile-midpoint target encoding with explicit response-scale semantics.

Adapted from pjankiewicz/mercari-solution, commit
374301ad1c32cbc93dcc40313d5d7bb9c5503746, mercari/regression_clf.py.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

This core has no price, currency or text semantics. Callers own the response
transform and fitting population. Soft precision has inverse squared response
units. Repeated quantiles are retained, rather than silently dropping classes.
"""
import numpy as np


def _vector(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in 'fiu' or values.ndim != 1 or not len(values):
        raise ValueError(name+' must be a nonempty real numeric vector')
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(name+' must be finite')
    return values


def quantile_boundaries(targets, n_bins):
    """Fit source-style linearly interpolated quantiles to caller training targets."""
    targets = _vector(targets, 'targets')
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError('Positive integer bin count required')
    levels = list(np.arange(0, 100, 100/n_bins))+[100]
    if len(levels) != n_bins+1:
        raise ValueError('Floating percentile grid does not produce the requested class count')
    return np.percentile(targets, levels, method='linear')


def quantile_midpoints(boundaries):
    boundaries = _vector(boundaries, 'boundaries')
    if len(boundaries) < 2 or (np.diff(boundaries) < 0).any():
        raise ValueError('At least two nondecreasing boundaries required')
    with np.errstate(over='ignore', invalid='ignore'):
        centers = np.mean([boundaries[:-1], boundaries[1:]], axis=0)
    if not np.isfinite(centers).all():
        raise ValueError('Boundary midpoints overflow')
    return centers


def encode_quantile_targets(targets, boundaries, *, soft=True, precision=160.0):
    """Encode targets; hard intervals are left-open/right-closed, including at min.

The source's direct exponential normalization is preserved. Complete underflow
is rejected instead of returning NaN or silently using a different formula.
Hard mode can return all-zero rows for the minimum or out-of-range targets.
"""
    targets = _vector(targets, 'targets')
    boundaries = _vector(boundaries, 'boundaries')
    centers = quantile_midpoints(boundaries)
    if type(soft) is not bool:
        raise ValueError('Explicit boolean soft mode required')
    if not soft:
        return ((targets[:, None] > boundaries[:-1]) &
                (targets[:, None] <= boundaries[1:])).astype(np.float32)
    if type(precision) not in (float, int) or not np.isfinite(precision) or precision <= 0:
        raise ValueError('Finite positive precision required')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        weights = np.exp(-0.5*precision*((targets[:, None]-centers)**2))
        encoded = (weights.T/weights.sum(axis=1)).T
    if not np.isfinite(encoded).all():
        raise ValueError('Soft target normalization underflow or overflow; revise response scale')
    return encoded


def decode_quantile_probabilities(probabilities, boundaries):
    """Return the midpoint expectation in the same units as the boundaries."""
    centers = quantile_midpoints(boundaries)
    probabilities = np.asarray(probabilities)
    if (probabilities.dtype.kind not in 'fiu' or probabilities.ndim != 2
            or probabilities.shape[1] != len(centers) or not len(probabilities)):
        raise ValueError('Nonempty aligned probability matrix required')
    probabilities = probabilities.astype(np.float64)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-8)):
        raise ValueError('Finite nonnegative normalized probabilities required')
    return (centers*probabilities).sum(axis=1)

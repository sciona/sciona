"""Domain-independent fixed-sum projection and periodic row smoothing.

No units, grid spacing, target meaning or population policy are inferred.
These operations do not claim to integrate a differential equation.
"""
import numpy as np


def _real_array(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError('Real numeric arrays required')
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Finite values required')
    return result


def project_nonnegative_fixed_sum(values, totals):
    """Return each row's Euclidean projection onto x>=0, sum(x)=total.

    Accept a nonempty 2-D real array and one finite nonnegative total per
    row. Return a fresh float64 array. Reject numerically unrepresentable
    projections; fixed-sum agreement uses rtol=1e-10 and atol=1e-12.
    """
    rows = _real_array(values)
    sums = _real_array(totals)
    if rows.ndim != 2 or not all(rows.shape) or sums.shape != (rows.shape[0],) or (sums < 0).any():
        raise ValueError('Nonempty rows and one nonnegative total per row required')
    result = np.zeros_like(rows)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for index, (row, total) in enumerate(zip(rows, sums)):
                if total == 0:
                    continue
                shifted = row - row.max()
                ordered = np.sort(shifted)[::-1]
                thresholds = (np.cumsum(ordered) - total) / np.arange(1, len(row) + 1)
                active = np.flatnonzero(ordered > thresholds)
                if not len(active):
                    raise ValueError('Unrepresentable projection')
                projected = np.maximum(shifted - thresholds[active[-1]], 0.)
                if not np.isfinite(projected).all() or not np.isclose(projected.sum(), total, rtol=1e-10, atol=1e-12):
                    raise ValueError('Numerically unstable fixed-sum projection')
                result[index] = projected
    except FloatingPointError as error:
        raise ValueError('Unrepresentable projection arithmetic') from error
    return result


def smooth_periodic_rows(values, weight):
    """Convex circular stencil [weight, 1-2*weight, weight] along rows.

    Nonempty 2-D finite real input; explicit finite scalar weight in [0,.5].
    Signed signals are allowed. The operation preserves each row sum and
    nonnegativity of nonnegative input up to floating-point rounding. It
    returns a fresh float64 array and does not infer physical boundaries.
    """
    rows = _real_array(values)
    if rows.ndim != 2 or not all(rows.shape):
        raise ValueError('Nonempty rows required')
    if type(weight) not in (int, float) or not 0 <= weight <= .5:
        raise ValueError('Weight must lie in [0,0.5]')
    # Weight before summing, so two large neighbors need not overflow first.
    try:
        with np.errstate(over='raise', invalid='raise'):
            result = (1 - 2 * weight) * rows + weight * np.roll(rows, 1, axis=1) + weight * np.roll(rows, -1, axis=1)
    except FloatingPointError as error:
        raise ValueError('Unrepresentable smoothing arithmetic') from error
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite smoothed values')
    return result

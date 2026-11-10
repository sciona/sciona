"""Uncentered affine least-squares fitting with an explicit rank cutoff."""
import numpy as np


def fit(coordinates, values, *, rcond):
    """Fit value = slope * coordinate + intercept, returning rank diagnostics.

Inputs are aligned finite real vectors in caller-owned units. Rank-deficient
and single-observation systems retain NumPy's minimum-norm solution; rank two
is required to identify both coefficients uniquely. No sorting, centering,
weighting, regularization or observation-availability inference is performed.
"""
    x,y=np.asarray(coordinates),np.asarray(values)
    if (x.ndim!=1 or not x.size or y.shape!=x.shape
            or x.dtype.kind not in 'iuf' or y.dtype.kind not in 'iuf'
            or x.dtype.itemsize>8 or y.dtype.itemsize>8
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Aligned nonempty finite real vectors required')
    if rcond is not None and (isinstance(rcond,(bool,np.bool_)) or not np.isscalar(rcond)
            or np.asarray(rcond).dtype.kind not in 'iuf'
            or not np.isrealobj(rcond) or not np.isfinite(rcond)
            or not (rcond == -1 or 0 <= rcond < 1)):
        raise ValueError('Explicit rcond must be None, -1 or a finite ratio in [0,1)')
    design=np.vstack([x,np.ones(len(x))]).T
    coefficients,residuals,rank,singular_values=np.linalg.lstsq(design,y,rcond=rcond)
    if not all(np.isfinite(a).all() for a in (coefficients,residuals,singular_values)):
        raise ValueError('Finite fitted state required')
    return dict(coefficients=coefficients,rank=int(rank),singular_values=singular_values,
                residual_sums_of_squares=residuals)

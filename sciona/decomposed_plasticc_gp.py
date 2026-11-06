"""Numeric GP conditioning boundaries, independent of astronomical metadata.

The caller supplies covariance blocks from one qualified kernel and invocation.
These operations do not choose units, fit hyperparameters, or authenticate that
provenance. The retained source fitter remains unchanged.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve


def _real(value):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError('Finite real numeric values required')
    return array.astype(np.float64, copy=True)


def scale_covariance(normalized, amplitude):
    """Scale a finite covariance block by an explicit positive variance."""
    matrix = _real(normalized)
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ValueError('Nonempty covariance matrix required')
    if isinstance(amplitude, (bool, np.bool_)) or not np.isscalar(amplitude):
        raise ValueError('Positive finite variance required')
    a = _real(amplitude)
    if a.ndim != 0 or a <= 0:
        raise ValueError('Positive finite variance required')
    result = matrix*float(a)
    if not np.isfinite(result).all():
        raise ValueError('Covariance scaling overflow')
    return result


def observation_variance(errors, white_noise_variance):
    """Combine explicit standard errors and white-noise variances.

    Preserve the retained solver's sqrt-then-square arithmetic, without adding
    any implicit noise floor. The caller supplies any model-default white noise.
    """
    error = _real(errors); white = _real(white_noise_variance)
    if error.ndim != 1 or not len(error) or white.shape != error.shape or np.any(error < 0) or np.any(white < 0):
        raise ValueError('Aligned nonnegative errors and white-noise variances required')
    result = np.sqrt(error**2 + white)**2
    if not np.isfinite(result).all():
        raise ValueError('Observation variance overflow')
    return result


def condition(covariance, targets, noise_variance):
    """Condition a zero-mean GP on supplied numeric covariance and diagonal noise.

    Require exact symmetry, then add nonnegative observation variance and use
    an upper Cholesky factor. No jitter or singular-matrix repair is introduced.
    Return independent factor, precision-weighted targets and log determinant.
    """
    matrix = _real(covariance); y = _real(targets); noise = _real(noise_variance)
    if matrix.ndim != 2 or not len(matrix) or matrix.shape != (len(matrix), len(matrix)):
        raise ValueError('Nonempty square covariance required')
    if not np.array_equal(matrix, matrix.T):
        raise ValueError('Symmetric covariance required')
    if y.shape != (len(matrix),) or noise.shape != y.shape or np.any(noise < 0):
        raise ValueError('Aligned targets and nonnegative noise variances required')
    matrix[np.diag_indices_from(matrix)] += noise
    if not np.isfinite(matrix).all():
        raise ValueError('Conditioning covariance overflow')
    factor = cholesky(matrix, lower=False)
    alpha = cho_solve((factor, False), y)
    logdet = float(2*np.sum(np.log(np.diag(factor))))
    if not np.isfinite(alpha).all() or not np.isfinite(logdet):
        raise ValueError('Nonfinite conditioned state')
    return factor, alpha, logdet


def predict(factor, alpha, cross_covariance, prior_variance):
    """Return zero-mean conditional means and latent marginal variances.

    Cross covariance is query-by-training; prior variance is query-aligned.
    Roundoff-negative posterior variance is retained, without clipping. Inputs
    are checked structurally but must belong to the same conditioned model.
    """
    upper = _real(factor); weights = _real(alpha)
    cross = _real(cross_covariance); diagonal = _real(prior_variance)
    if (upper.ndim != 2 or not len(upper) or upper.shape != (len(upper), len(upper))
            or np.any(np.diag(upper) <= 0) or np.any(np.tril(upper, -1) != 0)):
        raise ValueError('Positive-diagonal upper triangular factor required')
    if weights.shape != (len(upper),) or cross.ndim != 2 or not len(cross) or cross.shape[1] != len(upper):
        raise ValueError('Aligned weights and nonempty query-by-training covariance required')
    if diagonal.shape != (len(cross),) or np.any(diagonal < 0):
        raise ValueError('Aligned nonnegative prior variance required')
    mean = np.dot(cross, weights)
    inverse_cross = cho_solve((upper, False), cross.T)
    variance = diagonal - np.sum(cross.T*inverse_cross, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError('Nonfinite conditional output')
    return mean, variance

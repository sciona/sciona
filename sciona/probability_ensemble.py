"""Draft reusable numerical operations for aligned categorical ensembles.

Rows and class columns must have the same caller-established meaning across
inputs. Numeric validation cannot infer semantic alignment or calibration.
These functions neither fit models nor establish their empirical usefulness.
"""
import numpy as np


def _probabilities(values, ndim):
    source = np.asarray(values)
    if source.dtype.kind not in 'fiu' or source.ndim != ndim:
        raise ValueError('Real probability arrays required')
    array = np.asarray(source, dtype=np.float64)
    if any(size == 0 for size in array.shape) or array.shape[-1] < 2:
        raise ValueError('Nonempty rows and at least two classes required')
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError('Finite probabilities in [0, 1] required')
    if not np.allclose(array.sum(axis=-1), 1., rtol=0, atol=1e-8):
        raise ValueError('Normalized categorical probabilities required')
    return array


def mean_probability_bag(values):
    """Mean over runs in a (runs, rows, classes) float64 probability array.

Run counts and class counts are unrestricted beyond nonempty runs and at
least two classes. Source-specific counts belong in the calling adapter.
Inputs are converted to float64 and never modified.
"""
    return _probabilities(values, 3).mean(axis=0)


def geometric_arithmetic_probability_blend(left, right, additive,
                                           geometric_weight, additive_weight,
                                           left_exponent):
    """Combine aligned row/class matrices, then normalize the final sum.

raw = w * left**a * right**(1-a) + v * additive.
w, v and a are explicit real scalars in [0,1]; w + v must equal one within
1e-15 absolute tolerance. Endpoints select their branch
exactly, including at zero probability. Zero total support is rejected.
Tie-breaking selects the lowest class-column index, not a domain label.
"""
    arrays = [_probabilities(x, 2) for x in (left, right, additive)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError('Aligned probability shapes required')
    for value in (geometric_weight, additive_weight, left_exponent):
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Finite real scalar weights in [0, 1] required')
    if abs(geometric_weight + additive_weight - 1) > 1e-15:
        raise ValueError('Mixture weights must sum to one')
    left, right, additive = arrays
    if left_exponent == 0:
        geometric = geometric_weight * right
    elif left_exponent == 1:
        geometric = geometric_weight * left
    else:
        # Preserve left-to-right float64 arithmetic of the source expression.
        geometric = geometric_weight * np.power(left, left_exponent) * np.power(right, 1-left_exponent)
    raw = geometric + additive_weight * additive
    total = raw.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError('Positive combined probability support required')
    probability = raw / total
    return dict(raw_scores=raw.tolist(), probabilities=probability.tolist(),
                classes=np.argmax(probability, axis=1).tolist())

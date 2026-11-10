"""OSIC source prediction policies over explicit caller-aligned numeric arrays.

Application adapters own identity joins, baseline selection and which outcomes
are already known at prediction time. No clipping or calibration is inferred.
The final linear combination reuses an existing domain-independent operation.
"""
import numpy as np
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum


def _numeric(value, shape=None):
    if (not isinstance(value, np.ndarray) or value.dtype.kind not in 'iuf'
            or not value.size or (shape is not None and value.shape != shape)
            or not np.isfinite(value).all()):
        raise ValueError('Nonempty finite real arrays with explicit matching shapes required')
    result = value.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Float64 representation required')
    return result


def image_trajectories(slice_slopes, baseline_time, baseline_value,
                       baseline_uncertainty, query_subject, query_time):
    """Median per-subject slice slopes, then source intercept and extrapolation.

Require at least two slice predictions per subject, one baseline per subject,
and integer subject positions for each query. Time and slope coordinates must
agree. Uncertainty is the source heuristic, not a calibrated interval; negative
results are preserved. Source arithmetic forms the intercept before prediction.
"""
    if not isinstance(slice_slopes, (list, tuple)) or not slice_slopes:
        raise ValueError('Explicit per-subject slice predictions required')
    slopes = []
    for values in slice_slopes:
        values = _numeric(values)
        if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1) or len(values) < 2:
            raise ValueError('At least two scalar slice predictions per subject required')
        slopes.append(np.quantile(values, .5))
    a = np.asarray(slopes)
    bt, bv, bu = [_numeric(v, a.shape) for v in (baseline_time, baseline_value, baseline_uncertainty)]
    qt = _numeric(query_time)
    if (qt.ndim != 1 or not isinstance(query_subject, np.ndarray)
            or query_subject.shape != qt.shape or query_subject.dtype.kind not in 'iu'
            or np.any(query_subject >= len(a)) or np.any(query_subject < 0)):
        raise ValueError('One valid integer subject position per query required')
    index = query_subject.astype(np.intp)
    with np.errstate(over='raise', invalid='raise'):
        intercept = bv - a * bt
        point = a[index] * qt + intercept[index]
        uncertainty = bu[index] - a[index] * np.abs(bt[index] - qt)
    return np.column_stack((point, uncertainty))


def quantile_predictions(folds, observed_mask, observed_values):
    """Average fold predictions in source order, extract midpoint and raw spread.

Inputs are folds-by-queries-by-three output heads. Crossings and negative spread
are retained. The boolean mask identifies outcomes already known to the caller;
only those rows are replaced with known values and source confidence 0.1. Apply
this policy before final blending. It is unsuitable for unknown future outcomes.
"""
    _numeric(folds)
    if folds.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Source prediction outputs must be float32 or float64')
    # The notebook divides each model output in its original dtype before
    # accumulating into float64. Casting before division changes rounding.
    values = folds
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError('Fold predictions must have three output heads')
    n = values.shape[1]
    known = _numeric(observed_values, (n,))
    if not isinstance(observed_mask, np.ndarray) or observed_mask.dtype != np.bool_ or observed_mask.shape != (n,):
        raise ValueError('Boolean observation-availability mask required')
    average = np.zeros((n, 3))
    with np.errstate(over='raise', invalid='raise'):
        for prediction in values:
            average += prediction / len(values)
        result = np.column_stack((1.0 * average[:, 1], average[:, 2] - average[:, 0]))
    result[observed_mask, 0] = known[observed_mask]
    result[observed_mask, 1] = .1
    return result


def blend_predictions(image_keys, image_values, quantile_keys, quantile_values):
    """Sort both unique key sets, align them and apply source weights 0.45/0.55.

Each numeric matrix has point and uncertainty columns. Returns sorted keys and
a fresh matrix. Key creation and common units are caller responsibilities. This
source recipe uses the registered generic ordered weighted-sum operation.
"""
    for keys in (image_keys, quantile_keys):
        if (not isinstance(keys, (tuple, list)) or not keys
                or any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys)):
            raise ValueError('Nonempty unique string query keys required')
    if set(image_keys) != set(quantile_keys):
        raise ValueError('Prediction branches must have identical query keys')
    first = _numeric(image_values, (len(image_keys), 2))
    second = _numeric(quantile_values, (len(quantile_keys), 2))
    keys = sorted(image_keys)
    left = {key: i for i, key in enumerate(image_keys)}
    right = {key: i for i, key in enumerate(quantile_keys)}
    rows = np.stack((first[[left[k] for k in keys]].ravel(), second[[right[k] for k in keys]].ravel()))
    return keys, ordered_weighted_sum(rows, np.array([.45, .55])).reshape(-1, 2)

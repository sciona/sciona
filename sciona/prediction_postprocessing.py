"""Domain-independent bounded integer predictions with explicit offset semantics."""
import numpy as np


def clip_truncate_prediction(values, offset, lower, upper):
    """Add aligned offsets, clip, then truncate toward zero to signed int64.

    Inputs are equal-shaped finite real arrays; offsets share prediction units.
    Bounds are exactly representable integers within the float64 exact-integer
    range. This operation does not infer calibration, units or population scope.
    Caller data are not mutated. Zero-sized arrays are supported.
    """
    if type(lower) is not int or type(upper) is not int or not -(2**53-1) <= lower <= upper <= 2**53-1:
        raise ValueError('Ordered exactly representable integer bounds required')
    converted = []
    for value in (values, offset):
        if not isinstance(value, np.ndarray) or value.dtype.kind not in 'iuf':
            raise ValueError('Real numeric arrays required')
        array = value.astype(np.float64, copy=True)
        if not np.isfinite(array).all():
            raise ValueError('Finite arrays required')
        converted.append(array)
    if converted[0].shape != converted[1].shape:
        raise ValueError('Aligned shapes required; broadcasting is not supported')
    with np.errstate(over='ignore', invalid='ignore'):
        adjusted = converted[0] + converted[1]
    if not np.isfinite(adjusted).all():
        raise ValueError('Nonfinite offset-adjusted prediction')
    return np.clip(adjusted, lower, upper).astype(np.int64)

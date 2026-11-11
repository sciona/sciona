"""Independent TGS mosaic boundary features, before nearest-neighbor assembly.

Reference: Generate_Mosaic.R at winner commit
2f81d4dd8d50a01579e5f7650259dde92c5c3b8d. Input axes are population,
x, y as used by imager; callers with row/column arrays must transpose them.
"""
import numpy as np


def _sample_scale(values, axis):
    finite = np.isfinite(values)
    count = finite.sum(axis=axis, keepdims=True)
    total = np.where(finite, values, 0).sum(axis=axis, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        centered = values - total / count
        squares = np.where(finite, centered * centered, 0).sum(axis=axis, keepdims=True)
        # R scale uses max(1, observed_count - 1) for the divisor.
        return centered / np.sqrt(squares / np.maximum(1, count - 1))


def boundary_features(images):
    """Extrapolate each boundary, population-scale, then within-image-scale.

    Returns source direction names u/d (x boundaries) and l/r (y boundaries).
    Missing values caused by constant dimensions are zeroed AFTER both scales,
    matching the source. Features depend on the entire input population.
    """
    values = np.asarray(images, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError('images require finite population/x/y axes of size at least two')
    with np.errstate(over='ignore', invalid='ignore'):
        edges = dict(u=2 * values[:, 0, :] - values[:, 1, :],
                     d=2 * values[:, -1, :] - values[:, -2, :],
                     l=2 * values[:, :, 0] - values[:, :, 1],
                     r=2 * values[:, :, -1] - values[:, :, -2])
    result = {}
    for name, edge in edges.items():
        if not np.isfinite(edge).all():
            raise ValueError('boundary extrapolation overflow')
        with np.errstate(over='raise'):
            scaled = _sample_scale(_sample_scale(edge, 0), 1)
        result[name] = np.nan_to_num(scaled, nan=0., posinf=0., neginf=0.)
    return result

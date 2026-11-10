"""Training-fitted nonnegative clipping for generic sparse feature matrices.

Adapted from mercari/transformers.py in pjankiewicz/mercari-solution,
commit 374301ad1c32cbc93dcc40313d5d7bb9c5503746.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

The caller owns feature meanings, ordering, units and the training population.
This operation does not make arbitrary signed features semantically suitable
for nonnegative models. NaNs become zero; infinities at inference are clipped.
Fit uses stored entries, exactly as the source does, excluding implicit zeros.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse


def _matrix(values):
    if not hasattr(values, 'dtype'):
        values = np.asarray(values)
    if values.dtype.kind not in 'biuf' or values.ndim != 2:
        raise ValueError('A two-dimensional real numeric matrix is required')
    matrix = sparse.csr_matrix(values, dtype=np.float32, copy=True)
    if matrix.shape[1] == 0:
        raise ValueError('At least one feature is required')
    if not matrix.has_canonical_format:
        raise ValueError('Canonical sparse storage is required; resolve duplicate entries upstream')
    return matrix


@dataclass(frozen=True)
class FittedSparseClip:
    """Immutable training state; no dataset names or feature labels are stored."""

    n_features: int
    upper: float

    def __post_init__(self):
        if type(self.n_features) is not int or self.n_features < 1:
            raise ValueError('Positive integer feature width required')
        if not np.isscalar(self.upper) or not np.isfinite(self.upper) or self.upper < 0:
            raise ValueError('Finite nonnegative training bound required')
        if self.upper > np.finfo(np.float32).max:
            raise ValueError('Training bound exceeds float32 range')

    def transform(self, values):
        """Return an independent float32 CSR result using frozen training bounds."""
        matrix = _matrix(values)
        if matrix.shape[1] != self.n_features:
            raise ValueError('Feature width differs from fitted training state')
        matrix.data[np.isnan(matrix.data)] = 0
        matrix.data = matrix.data.clip(0, self.upper)
        return matrix


def fit_sparse_clip(training):
    """Fit on training only; reject undefined/nonfinite or negative source bounds.

Explicit zeros are retained. A matrix with no stored entries or only NaNs has
no source-style fitted maximum and is rejected, rather than inventing a bound.
"""
    matrix = _matrix(training)
    if matrix.shape[0] == 0 or not matrix.data.size or np.isnan(matrix.data).all():
        raise ValueError('Training requires at least one stored non-NaN value')
    if np.isinf(matrix.data).any():
        raise ValueError('Infinite training entries are unsupported')
    return FittedSparseClip(matrix.shape[1], float(np.nanmax(matrix.data)))

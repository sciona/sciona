"""Source recipe adapters for explicitly bound reusable circular filters."""
import os
from pathlib import Path
import numpy as np
from sciona.connectomics_runtime import Prepared, prepare
from sciona.connectomics_source import source_namespace


def filter_configuration(prepared):
    if not isinstance(prepared, Prepared):
        raise ValueError('Prepared Connectomics input required')
    prepared = prepare(dict(version=1, signals=prepared.signals,
                            mode=prepared.mode, directivity=prepared.directivity))
    return (prepared.signals, 0,
            [0, -1, 1], [1., 1., 1.],
            [0, 1, 2, 3], [1., 1., .8, .4],
            [0, -1, -2, 1], [1., 1., 1., 1.],
            [0, -1, -2, -3], [1., 1., 1., 1.])


def checked_filter(signals, filtered):
    """Bind a finite prefiltered matrix to one exact source input.

    Correct tap computation is the graph's responsibility; this adapter checks
    the array contract and prevents reuse with different source signals.
    """
    if (not isinstance(filtered, np.ndarray) or filtered.dtype != np.float32
            or filtered.shape != signals.shape or not np.isfinite(filtered).all()):
        raise ValueError('Finite float32 prefiltered matrix matching signals required')
    snapshot = filtered.copy(order='K')
    expected = signals.copy(order='K')
    def apply(values):
        if not np.array_equal(values, expected):
            raise ValueError('Prefiltered matrix bound to different source signals')
        return snapshot.copy(order='K')
    return apply


def execute_prefiltered(prepared, f1, f2, f3, f4):
    if not isinstance(prepared, Prepared):
        raise ValueError('Prepared Connectomics input required')
    # Revalidate even a manually constructed or subsequently modified object.
    prepared = prepare(dict(version=1, signals=prepared.signals,
                            mode=prepared.mode, directivity=prepared.directivity))
    directory = os.environ.get('SCIONA_CONNECTOMICS_SOURCE_DIR')
    if not directory:
        raise ValueError('Provision SCIONA_CONNECTOMICS_SOURCE_DIR before execution')
    ns = source_namespace(Path(directory))
    for name, values in zip(('f1', 'f2', 'f3', 'f4'), (f1, f2, f3, f4), strict=True):
        ns[name] = checked_filter(prepared.signals, values)
    historical = ns['PCA']

    class CheckedPCA(historical):
        def fit(self, values, *args, **kwargs):
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite filtered signals')
            result = super().fit(values, *args, **kwargs)
            if not np.isfinite(self.noise_variance_) or self.noise_variance_ <= 0:
                raise ValueError('Degenerate residual covariance')
            return result

        def get_precision(self):
            result = super().get_precision()
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite historical precision')
            return result

    original_scale = ns['scale']

    def checked_scale(values):
        diagonal_min = values.copy()
        np.fill_diagonal(diagonal_min, diagonal_min.min())
        if not np.isfinite(diagonal_min).all() or np.ptp(diagonal_min) <= 0:
            raise ValueError('Degenerate score normalization')
        result = original_scale(values)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite normalized scores')
        return result

    ns.update(PCA=CheckedPCA, scale=checked_scale, print=lambda *a, **kw: None)
    scores = ns['make_' + prepared.mode + '_inference'](prepared.signals)
    if prepared.directivity:
        directed = ns['make_prediction_directivity'](prepared.signals, n_jobs=1)
        scores = .997 * scores + .003 * directed
    return dict(version=1, mode=prepared.mode, directivity=prepared.directivity,
                pca_fits=240 if prepared.mode == 'simple' else 480,
                scores=scores.tolist())

"""Synthetic MeanShift control semantics against the installed float64 backend."""
import numpy as np
import pytest
from sklearn.cluster import MeanShift
from threadpoolctl import threadpool_limits
from sciona.atoms.ml.sklearn.cluster.atoms import mean_shift


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('mode', ['rows', 'bins', 'explicit'])
@pytest.mark.parametrize('cluster_all', [False, True])
@pytest.mark.parametrize('max_iter', [0, 300])
def test_controls(dtype, mode, cluster_all, max_iter):
    points = np.array([[0., 0.], [.125, 0.], [0., .125],
                       [3., 3.], [3.125, 3.], [3., 3.125], [10., 10.]], dtype=dtype)
    seeds = np.array([[0., 0.], [3., 3.]], dtype=dtype) if mode == 'explicit' else None
    options = dict(bandwidth=.5, seeds=seeds, bin_seeding=mode == 'bins',
                   min_bin_freq=2, cluster_all=cluster_all, max_iter=max_iter, n_jobs=1)
    before = points.copy()
    with threadpool_limits(limits=1):
        centers, labels = mean_shift(points, **options)
        reference_options = dict(options)
        if seeds is not None:
            reference_options['seeds'] = seeds.astype(np.float64)
        expected = MeanShift(**reference_options).fit(points.astype(np.float64))
    np.testing.assert_array_equal(centers, expected.cluster_centers_)
    np.testing.assert_array_equal(labels, expected.labels_)
    np.testing.assert_array_equal(points, before)
    assert centers.dtype == np.float64
    if mode != 'rows' and not cluster_all:
        assert labels[-1] == -1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_estimated_bandwidth(dtype):
    points = np.random.default_rng(42).normal(size=(30, 2)).astype(dtype)
    with threadpool_limits(limits=1):
        centers, labels = mean_shift(points, bandwidth=None, n_jobs=1)
        expected = MeanShift(n_jobs=1).fit(points.astype(np.float64))
    np.testing.assert_array_equal(centers, expected.cluster_centers_)
    np.testing.assert_array_equal(labels, expected.labels_)


def test_no_seed_has_neighbors():
    with threadpool_limits(limits=1), pytest.raises(ValueError, match='No point was within'):
        mean_shift(np.array([[0.], [1.]]), bandwidth=.1, seeds=np.array([[10.]]), n_jobs=1)


def test_translation():
    points = np.array([[0.], [.125], [8.], [8.125]])
    with threadpool_limits(limits=1):
        centers, labels = mean_shift(points, bandwidth=.5, n_jobs=1)
        shifted, shifted_labels = mean_shift(points + 16, bandwidth=.5, n_jobs=1)
    np.testing.assert_array_equal(centers + 16, shifted)
    np.testing.assert_array_equal(labels, shifted_labels)

"""Synthetic backend parity and boundary checks for reusable Lloyd fitting."""
import numpy as np
import pytest
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from sciona.atoms.ml.sklearn.cluster.lloyd_fit import fit_lloyd_clusters


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('dimensions', [1, 3])
@pytest.mark.parametrize('seed', [0, 1, 19])
@pytest.mark.parametrize('restarts', [1, 3])
def test_backend(dtype, dimensions, seed, restarts):
    points = np.random.default_rng(seed).normal(size=(30, dimensions)).astype(dtype)
    before = points.copy()
    centers, labels = fit_lloyd_clusters(points, 4, seed, restarts)
    with threadpool_limits(limits=1):
        expected = KMeans(n_clusters=4, random_state=seed, n_init=restarts).fit(points)
    np.testing.assert_array_equal(centers, expected.cluster_centers_)
    np.testing.assert_array_equal(labels, expected.labels_)
    np.testing.assert_array_equal(points, before)
    assert centers.dtype == dtype
    assert not np.shares_memory(centers, points)


@pytest.mark.parametrize('kwargs', [dict(n_clusters=0), dict(n_clusters=4),
    dict(n_clusters=True), dict(random_state=-1), dict(random_state=2**32),
    dict(n_init=0), dict(max_iter=False), dict(tolerance=-1.), dict(tolerance=np.nan)])
def test_controls(kwargs):
    options = dict(n_clusters=2)
    options.update(kwargs)
    with pytest.raises(ValueError):
        fit_lloyd_clusters(np.ones((3, 2)), **options)


@pytest.mark.parametrize('points', [np.ones((3, 2), dtype=int), np.ones(3),
    np.empty((0, 2)), np.empty((3, 0)), np.array([[np.nan], [1.]]),
    np.array([[np.inf], [1.]])])
def test_inputs(points):
    with pytest.raises(ValueError):
        fit_lloyd_clusters(points, 1)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('iterations', [1, 2, 300])
@pytest.mark.parametrize('tolerance', [0., .25])
def test_convergence_controls(dtype, iterations, tolerance):
    points = np.random.default_rng(72).normal(size=(40, 3)).astype(dtype)
    centers, labels = fit_lloyd_clusters(points, 5, 7, 2, iterations, tolerance)
    with threadpool_limits(limits=1):
        expected = KMeans(n_clusters=5, random_state=7, n_init=2,
                          max_iter=iterations, tol=tolerance).fit(points)
    np.testing.assert_array_equal(centers, expected.cluster_centers_)
    np.testing.assert_array_equal(labels, expected.labels_)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_duplicate_points(dtype):
    from sklearn.exceptions import ConvergenceWarning
    points = np.array([[0., 0.], [0., 0.], [2., 2.], [2., 2.]], dtype=dtype)
    with pytest.warns(ConvergenceWarning):
        centers, labels = fit_lloyd_clusters(points, 3)
    assert centers.shape == (3, 2)
    assert len(np.unique(labels)) == 2
    np.testing.assert_array_equal(centers[labels], points)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_feature_translation(dtype):
    points = np.array([[0., 0.], [0., 2.], [16., 0.], [16., 2.]], dtype=dtype)
    centers, labels = fit_lloyd_clusters(points, 2)
    shifted, shifted_labels = fit_lloyd_clusters(points + 32, 2)
    np.testing.assert_array_equal(labels, shifted_labels)
    np.testing.assert_array_equal(centers + 32, shifted)


@pytest.mark.parametrize('scale', [.8, 1., 1.3])
@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('seed', [0, 1])
def test_detector_clustering_injection(scale, paired, seed):
    import pandas as pd
    from types import SimpleNamespace
    from sciona.atoms.ml.sklearn.cluster.atoms import mean_shift
    from sciona.decomposed_trackml_lifecycle import _bound
    from sciona.trackml_detector import CapsSpec
    from scripts.validate_trackml_detector import synthetic_modules
    calls = []

    class KMeansAdapter:
        def __init__(self, *, n_clusters, n_init, random_state):
            self.options = dict(n_clusters=n_clusters, n_init=n_init, random_state=random_state)

        def fit(self, points):
            self.cluster_centers_, self.labels_ = fit_lloyd_clusters(points, **self.options)
            calls.append('kmeans')
            return self

    class MeanShiftAdapter:
        def __init__(self, *, bandwidth):
            self.bandwidth = bandwidth

        def fit(self, points):
            self.cluster_centers_, self.labels_ = mean_shift(points, bandwidth=self.bandwidth, n_jobs=1)
            calls.append('meanshift')
            return self

    class BoundCaps(CapsSpec):
        pass

    BoundCaps.__init__ = _bound(CapsSpec.__init__, dict(KMeans=KMeansAdapter, MeanShift=MeanShiftAdapter))
    points = synthetic_modules(scale, paired).sample(frac=1, random_state=seed)
    before = points.copy(deep=True)
    with threadpool_limits(limits=1):
        expected = CapsSpec(SimpleNamespace(detectors_df=points))
        actual = BoundCaps(SimpleNamespace(detectors_df=points))
    assert calls == ['kmeans', 'kmeans', 'meanshift']
    assert vars(actual).keys() == vars(expected).keys()
    for name, value in vars(expected).items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, getattr(actual, name), check_exact=True)
        else:
            np.testing.assert_array_equal(value, getattr(actual, name))
    pd.testing.assert_frame_equal(points, before, check_exact=True)

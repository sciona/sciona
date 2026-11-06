"""Synthetic backend parity and independent contracts for reusable search."""
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from sciona.atoms.ml.sklearn.neighbors.fitted_search import (
    fit_euclidean_search as fit, query_euclidean_k as knn,
    query_euclidean_radius as radius, concrete_only,
)


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('samples', [6, 16, 64])
def test_exact_backend_distances_and_order(seed, dimension, samples):
    rng = np.random.RandomState(seed+915)
    points, query = rng.normal(size=(samples, dimension)), rng.normal(size=(7, dimension))
    saved = points.copy()
    with threadpool_limits(limits=1):
        reference = NearestNeighbors(n_jobs=1).fit(points)
        state = fit(points)
        for actual, expected in [(knn(query, state, 3), reference.kneighbors(query, n_neighbors=3)),
                                 (radius(query, state, 1.5), reference.radius_neighbors(query, radius=1.5))]:
            for rows, wanted in zip(actual, expected, strict=True):
                for row, expected_row in zip(rows, wanted, strict=True):
                    np.testing.assert_array_equal(row, expected_row)
    np.testing.assert_array_equal(points, saved)


def test_ties_preserve_backend_order():
    points = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [2., 0.], [-2., 0.]])
    query = np.zeros((1, 2))
    with threadpool_limits(limits=1):
        reference = NearestNeighbors(n_jobs=1).fit(points).kneighbors(query, n_neighbors=3)
        result = knn(query, fit(points), 3)
    for a, b in zip(result, reference, strict=True): np.testing.assert_array_equal(a, b)


def test_radius_boundary_zero_and_misses():
    points = np.array([[0., 0.], [3., 4.], [6., 8.]])
    state = fit(points)
    distances, indices = radius(points[:1], state, 5.)
    np.testing.assert_array_equal(indices[0], [0, 1])
    np.testing.assert_array_equal(distances[0], [0., 5.])
    distances, indices = radius(points[:1], state, 0.)
    np.testing.assert_array_equal(indices[0], [0])
    assert len(radius(np.array([[100., 100.]]), state, 1.)[0][0]) == 0


def test_fit_owns_storage_and_small_fit_can_query_one():
    points = np.array([[3., 4.]])
    state = fit(points)
    points[:] = 100.
    distances, indices = knn(np.zeros((1, 2)), state, 1)
    np.testing.assert_array_equal(distances, [[5.]])
    np.testing.assert_array_equal(indices, [[0]])
    with pytest.raises(ValueError): knn(points, state, 2)


@pytest.mark.parametrize('points', [np.zeros((0, 2)), np.zeros((2, 0)), np.zeros(2),
    np.zeros((2, 2), dtype=np.float32), np.array([[np.nan]]), np.array([[np.inf]])])
def test_invalid_fit_rejects(points):
    with pytest.raises(ValueError): fit(points)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_k_rejects(value):
    with pytest.raises(ValueError): knn(np.zeros((1, 2)), fit(np.ones((6, 2))), value)


@pytest.mark.parametrize('value', [-1., np.inf, np.nan, True])
def test_invalid_radius_rejects(value):
    with pytest.raises(ValueError): radius(np.zeros((1, 2)), fit(np.ones((6, 2))), value)


def test_concrete_only_and_query_feature_contract():
    with pytest.raises(ValueError): concrete_only()
    with pytest.raises(ValueError): knn(np.zeros((1, 3)), fit(np.ones((6, 2))), 1)
    with pytest.raises(ValueError): knn(np.zeros((0, 2)), fit(np.ones((6, 2))), 1)

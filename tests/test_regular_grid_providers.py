"""Synthetic backend parity and independent multilinear interpolation checks."""
import itertools
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from sciona.atoms.physics.regular_grid import fit_linear_grid, evaluate_linear_grid


@pytest.mark.parametrize('dimensions', [1, 2, 3])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('components', [(), (2,), (2, 2)])
def test_backend_parity_and_storage(dimensions, descending, components):
    rng = np.random.default_rng(47)
    axes = [np.linspace(-2., 3., 4+i) for i in range(dimensions)]
    if descending:
        axes = [a[::-1].copy() for a in axes]
    values = rng.normal(size=tuple(len(a) for a in axes)+components)
    points = rng.uniform(-4., 5., (27, dimensions))
    before = points.copy()
    reference = RegularGridInterpolator(axes, values, method='linear', bounds_error=False, fill_value=None)
    expected = reference(points)
    state = fit_linear_grid(axes, values)
    np.testing.assert_array_equal(evaluate_linear_grid(points, state), expected)
    assert evaluate_linear_grid(np.empty((0, dimensions)), state).shape == (0,)+components
    values[...] = 99.
    for a in axes:
        a[...] = 42.
    np.testing.assert_array_equal(evaluate_linear_grid(points, state), expected)
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize('dimensions', [1, 2, 3])
@pytest.mark.parametrize('descending', [False, True])
def test_independent_multilinear_polynomial(dimensions, descending):
    axes = [np.array([-2., -.5, 1., 3.]) for _ in range(dimensions)]
    if descending:
        axes[0] = axes[0][::-1]
    def polynomial(points):
        result = np.full(points.shape[:-1], 1.25)
        for count in range(1, dimensions+1):
            for subset in itertools.combinations(range(dimensions), count):
                result += (count+.5)*np.prod(points[..., list(subset)], axis=-1)
        return result
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    queries = np.array(list(itertools.product([-4., -2., .2, 3., 5.], repeat=dimensions)))
    state = fit_linear_grid(axes, polynomial(grid))
    np.testing.assert_allclose(evaluate_linear_grid(queries, state), polynomial(queries), rtol=3e-14, atol=3e-13)


@pytest.mark.parametrize('axis', [np.array([1.]), np.array([0., 0.]), np.array([0., 2., 1.]),
                               np.array([0., np.nan]), np.array([0., 1.], dtype=np.float32)])
def test_invalid_axes(axis):
    with pytest.raises(ValueError):
        fit_linear_grid([axis], np.ones(len(axis)))


@pytest.mark.parametrize('values', [np.ones(3), np.ones(2, dtype=np.float32), np.array([0., np.inf])])
def test_invalid_values(values):
    with pytest.raises(ValueError):
        fit_linear_grid([np.array([0., 1.])], values)


@pytest.mark.parametrize('points', [np.ones(2), np.ones((2, 2)), np.ones((2, 1), dtype=np.float32), np.array([[np.nan]])])
def test_invalid_queries(points):
    state = fit_linear_grid([np.array([0., 1.])], np.array([0., 1.]))
    with pytest.raises(ValueError):
        evaluate_linear_grid(points, state)


def test_configuration_and_overflow_reject():
    state = fit_linear_grid([np.array([0., 1.])], np.array([-1e308, 1e308]))
    with pytest.raises(ValueError):
        evaluate_linear_grid(np.array([[3.]]), state)
    state.method = 'nearest'
    with pytest.raises(ValueError):
        evaluate_linear_grid(np.array([[.5]]), state)


def test_registered_abstract_witness_rejects():
    from sciona.ghost.registry import get_witness
    for function in (fit_linear_grid, evaluate_linear_grid):
        witness = get_witness(function.__module__+'.'+function.__name__)
        with pytest.raises(ValueError, match='Concrete grid'):
            witness()

"""Synthetic source parity, independent geometry and rejection boundaries."""
import numpy as np
import pytest

from sciona.atoms.physics.tangent_residuals import (
    hemisphere_tangent_displacement, combine_scaled_uncertainty,
)
from sciona.trackml_bayesian import BayesianNeighborScorer


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('hemisphere', [-1, 0, 1])
def test_projection_source_and_basis(seed, hemisphere):
    rng = np.random.default_rng(seed)
    p = rng.uniform(.5, 5., (31, 3))
    p[:, :2] *= rng.choice([-1, 1], size=(31, 2))
    p[:, 2] *= hemisphere
    q = p + rng.normal(size=p.shape)
    before = p.copy(), q.copy()
    actual = hemisphere_tangent_displacement(p, q)
    expected = BayesianNeighborScorer().projectNeighborDisplacement(*p.T, *q.T)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    theta = np.arccos(p[:, 2]/np.linalg.norm(p, axis=1))
    phi = np.arctan2(p[:, 1], p[:, 0])
    basis = np.column_stack([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    np.testing.assert_allclose(actual[0], hemisphere*np.sum(basis*(p-q), axis=1), atol=3e-14)
    np.testing.assert_array_equal(p, before[0])
    np.testing.assert_array_equal(q, before[1])


@pytest.mark.parametrize('seed', range(8))
def test_uncertainty_source_and_generic_coefficients(seed):
    rng = np.random.default_rng(seed)
    measured = rng.uniform(0., 5., (2, 31))
    driver = rng.uniform(-100., 100., 31)
    expected = BayesianNeighborScorer().predictRandomError(None, driver, *measured)
    for m, coefficient, e in zip(measured, [.2096, .3853], expected):
        np.testing.assert_array_equal(combine_scaled_uncertainty(m, driver, .007884, coefficient), e)
    np.testing.assert_allclose(combine_scaled_uncertainty(measured[0], driver, 2., 3.),
                               np.hypot(measured[0], 6.*driver), rtol=3e-15)


def test_empty_zero_and_axis():
    assert all(a.shape == (0,) for a in hemisphere_tangent_displacement(np.empty((0, 3)), np.empty((0, 3))))
    assert combine_scaled_uncertainty(np.empty(0), np.empty(0), 0., 0.).shape == (0,)
    np.testing.assert_array_equal(combine_scaled_uncertainty(np.zeros(2), np.ones(2), 0., 1.), np.zeros(2))
    with pytest.raises(ValueError):
        hemisphere_tangent_displacement(np.array([[0., 0., 1.]]), np.ones((1, 3)))


@pytest.mark.parametrize('bad', [np.ones((2, 2)), np.ones((2, 3), dtype=np.float32),
                              np.full((2, 3), np.nan), np.full((2, 3), 1e300), np.full((2, 3), 1e-300)])
def test_projection_rejects_invalid(bad):
    with pytest.raises(ValueError):
        hemisphere_tangent_displacement(bad, np.ones((2, 3)))


@pytest.mark.parametrize('coefficient', [True, -1., np.inf, np.nan, '1'])
def test_uncertainty_rejects_coefficients(coefficient):
    with pytest.raises(ValueError):
        combine_scaled_uncertainty(np.ones(2), np.ones(2), coefficient, 1.)
    with pytest.raises(ValueError):
        combine_scaled_uncertainty(np.ones(2), np.ones(2), 1., coefficient)


@pytest.mark.parametrize('measured,driver', [
    (np.ones(2, dtype=np.float32), np.ones(2)),
    (-np.ones(2), np.ones(2)), (np.ones(2), np.ones(3)),
    (np.ones(2), np.full(2, np.inf)), (np.full(2, 1e300), np.ones(2)),
])
def test_uncertainty_rejects_vectors(measured, driver):
    with pytest.raises(ValueError):
        combine_scaled_uncertainty(measured, driver, 1., 1.)

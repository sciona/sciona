"""Synthetic source-parity and independent projection invariants."""
from types import SimpleNamespace
import numpy as np
import pytest
from sciona.atoms.physics.coordinate_projection import cylindrical_projection, plane_projection
from sciona.trackml_neighbors import CylinderNeighbors, CapNeighbors


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('scale', [.01, 1., 100.])
def test_exact_source_and_projection_invariants(seed, scale):
    points = np.random.RandomState(seed).normal(size=(32, 3)) * scale
    saved = points.copy()
    radius, height, weight = 3. * scale, -2. * scale, .7
    cylinder = cylindrical_projection(points, radius, weight)
    expected = CylinderNeighbors.Cylinder.transformToNeighborCoords(
        SimpleNamespace(cyl_mean_r2=radius, zscale=weight), points)
    np.testing.assert_array_equal(cylinder, expected)
    np.testing.assert_allclose(np.linalg.norm(cylinder[:, :2], axis=1), radius*weight)
    plane = plane_projection(points, height)
    np.testing.assert_array_equal(plane, CapNeighbors.Cap.transformToNeighborCoords(SimpleNamespace(cap_z=height), points))
    np.testing.assert_allclose(plane[:, 0]*points[:, 2], points[:, 0]*height)
    np.testing.assert_allclose(plane[:, 1]*points[:, 2], points[:, 1]*height)
    np.testing.assert_array_equal(points, saved)


@pytest.mark.parametrize('operation,width', [(cylindrical_projection, 3), (plane_projection, 2)])
def test_empty_batches(operation, width):
    result = operation(np.empty((0, 3)), 2.)
    assert result.shape == (0, width) and result.dtype == np.float64


@pytest.mark.parametrize('operation,points', [
    (cylindrical_projection, [[0., 0., 1.]]),
    (plane_projection, [[1., 1., 0.]]),
    (cylindrical_projection, [[1e308, 1e308, 1.]]),
    (plane_projection, [[1e308, 1., 1e-308]]),
])
def test_undefined_geometry_rejects(operation, points):
    with pytest.raises(ValueError):
        operation(np.array(points), 2.)


@pytest.mark.parametrize('operation', [cylindrical_projection, plane_projection])
@pytest.mark.parametrize('value', [True, np.inf, np.nan])
def test_invalid_scalar(operation, value):
    with pytest.raises(ValueError):
        operation(np.ones((1, 3)), value)

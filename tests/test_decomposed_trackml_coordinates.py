"""Projection bindings must remain local to one lifecycle invocation."""
from types import SimpleNamespace
import numpy as np
import pytest
from sciona.decomposed_trackml_coordinate_lifecycle import neighbor_class, lifecycle_class
from sciona.trackml_neighbors import Neighbors, CylinderNeighbors, CapNeighbors
from sciona.trackml_runtime import TrackRuntime
from sciona.trackml_execution import CheckedLifecycle
from sciona.atoms.particle_tracking.helix_geometry import atoms


def test_neighbor_bindings_forward_parameters_without_mutating_source_classes():
    seen = []
    def cylinder(points, radius, weight):
        seen.append(('cylinder', radius, weight))
        return points.copy()
    def plane(points, height):
        seen.append(('plane', height))
        return points[:, :2].copy()
    first = neighbor_class(cylinder, plane)
    second = neighbor_class(lambda *a: np.zeros_like(a[0]), lambda *a: np.zeros((len(a[0]), 2)))
    spec = SimpleNamespace(caps=object(), cylinders=object())
    a, b = first(spec), second(spec)
    points = np.ones((2, 3))
    for neighbors in (a, b):
        c = neighbors.cyln.Cylinder.__new__(neighbors.cyln.Cylinder)
        c.cyl_mean_r2, c.zscale = 7., .4
        p = neighbors.capn.Cap.__new__(neighbors.capn.Cap)
        p.cap_z = -5.
        actual = (c.transformToNeighborCoords(points), p.transformToNeighborCoords(points))
        expected = (points, points[:, :2]) if neighbors is a else (np.zeros((2, 3)), np.zeros((2, 2)))
        for x, y in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(x, y)
    assert seen == [('cylinder', 7., .4), ('plane', -5.)]
    assert Neighbors.__init__.__globals__['CylinderNeighbors'] is CylinderNeighbors
    assert Neighbors.__init__.__globals__['CapNeighbors'] is CapNeighbors
    assert a.cyln.Cylinder is not b.cyln.Cylinder


def test_runtime_round_constructor_uses_local_neighbors_and_retains_population_checks():
    cls = lifecycle_class(atoms.circle_from_three_points, atoms.helix_pitch_from_two_points,
                          atoms.helix_pitch_least_squares, atoms.helix_nearest_point_distance,
                          lambda *a: a[0], lambda *a: a[0][:, :2])
    assert cls.Run.__init__.__code__ is TrackRuntime.Run.__init__.__code__
    assert cls.Run.__init__.__globals__['Neighbors'] is not Neighbors
    assert TrackRuntime.Run.__init__.__globals__['Neighbors'] is Neighbors
    assert cls.setupRun is CheckedLifecycle.setupRun


@pytest.mark.parametrize('providers', [(None, lambda *a: None), (lambda *a: None, None)])
def test_noncallable_projection_rejected(providers):
    with pytest.raises(ValueError, match='callable'):
        neighbor_class(*providers)

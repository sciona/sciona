"""Synthetic boundary cases for reuse of fixed-iteration helix projection."""
import numpy as np
import pytest
from sciona.atoms.particle_tracking.helix_geometry.atoms import helix_nearest_point_distance
from sciona.decomposed_trackml_nearest import nearest
from sciona.trackml_nearest import nearest_point_distance


def values():
    return [1., 0., 0., 0., 0., 1., 4.,
            np.array([1., 2.]), np.array([0., .5]), np.array([0., 1.])]


def test_empty_bypasses_nonempty_provider():
    def fail(*args, **kwargs):
        raise AssertionError('Empty batch must bypass provider')
    result = nearest(*[np.empty(0) for _ in range(10)], operation=fail)
    assert len(result) == 4
    assert all(a.shape == (0,) and a.dtype == np.float64 for a in result)


@pytest.mark.parametrize('iterations', [1, 3, 6])
def test_explicit_provider_and_iteration_forwarding(iterations):
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return helix_nearest_point_distance(*args, **kwargs)
    args = values()
    actual = nearest(*args, iterations=iterations, operation=observed)
    expected = nearest_point_distance(*args, iterations=iterations)
    assert calls == [dict(iterations=iterations)]
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('iterations', [True, 0, -1, 1.5])
def test_invalid_iterations(iterations):
    with pytest.raises(ValueError):
        nearest(*values(), iterations=iterations, operation=helix_nearest_point_distance)


@pytest.mark.parametrize('index,value', [(0, np.nan), (6, np.inf),
                                         (7, np.ones(3)), (8, np.ones(2, dtype=np.float32))])
def test_invalid_parameters(index, value):
    args = values()
    args[index] = value
    with pytest.raises(ValueError):
        nearest(*args, operation=helix_nearest_point_distance)


def test_nonfinite_output_rejected():
    def invalid(*args, **kwargs):
        return tuple(np.full(2, np.inf) for _ in range(4))
    with pytest.raises(ValueError, match='no finite result'):
        nearest(*values(), operation=invalid)


def test_noncallable_provider_rejected():
    with pytest.raises(ValueError, match='Callable'):
        nearest(*values(), operation=None)


def test_lifecycle_bindings_are_invocation_local():
    from sciona.atoms.particle_tracking.helix_geometry import atoms
    from sciona.decomposed_trackml_projection_lifecycle import lifecycle_class
    from sciona.trackml_extension import TrackExtension
    from sciona.trackml_ranking import RankedTrackExtension
    functions = (atoms.circle_from_three_points, atoms.helix_pitch_from_two_points,
                 atoms.helix_pitch_least_squares)
    original = (TrackExtension.findHelixIntersectionNeighbors,
                RankedTrackExtension.evaluateLocally)
    previous = [f.__globals__['helixNearestPointDistance'] for f in original]
    first = lifecycle_class(*functions, atoms.helix_nearest_point_distance)
    def other(*args, **kwargs):
        return atoms.helix_nearest_point_distance(*args, **kwargs)
    second = lifecycle_class(*functions, other)
    for name, source, saved in zip(('findHelixIntersectionNeighbors', 'evaluateLocally'),
                                    original, previous, strict=True):
        a, b = getattr(first, name), getattr(second, name)
        assert a.__code__ is source.__code__ and b.__code__ is source.__code__
        assert a.__globals__['helixNearestPointDistance'].keywords['operation'] is atoms.helix_nearest_point_distance
        assert b.__globals__['helixNearestPointDistance'].keywords['operation'] is other
        assert source.__globals__['helixNearestPointDistance'] is saved

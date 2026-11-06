"""Invocation isolation and empty selection at the intersection recipe boundary."""
from types import SimpleNamespace
import numpy as np
import pytest
from sciona.decomposed_trackml_intersections import intersector_class
from sciona.trackml_intersections import Intersector, CylinderIntersector, CapIntersector
from sciona.atoms.physics.helix_cylinder import helix_cylinder_intersection
from sciona.atoms.particle_tracking.track_matching.atoms import helix_cap_intersection


def spec():
    return SimpleNamespace(cylinders=SimpleNamespace(cyl_rsqr=np.array([4.])),
        caps=SimpleNamespace(cap_z=np.array([1.]), ring_overlap_r2_min_sqr=[], ring_overlap_r2_max_sqr=[]))


def test_independent_bindings_and_unchanged_source():
    calls = []
    def plane(*args):
        calls.append(True)
        return helix_cap_intersection(*args)
    a = intersector_class(helix_cylinder_intersection, plane)(spec())
    b = intersector_class(helix_cylinder_intersection, helix_cap_intersection)(spec())
    args = [np.array([v]) for v in (3., .5, 0., 1., 1., .5, 2., 3.)]
    first = a.capis.intersectHelices(*args, pre_move=0., missable=False)
    second = b.capis.intersectHelices(*args, pre_move=0., missable=False)
    for x, y in zip(first, second, strict=True):
        np.testing.assert_array_equal(x, y)
    assert calls == [True]
    assert type(a.capis) is not type(b.capis)
    assert Intersector.__init__.__globals__['CylinderIntersector'] is CylinderIntersector
    assert Intersector.__init__.__globals__['CapIntersector'] is CapIntersector


def test_empty_selected_plane_batch_does_not_call_nonempty_provider():
    def forbidden(*args):
        raise AssertionError('No selected intersections')
    instance = intersector_class(helix_cylinder_intersection, forbidden)(spec())
    args = [np.array([v]) for v in (3., .5, 2., 1., 1., .5, 2., 3.)]
    result = instance.capis.intersectHelices(*args, pre_move=0., missable=False)
    assert result[3][0] == -1 and np.isinf(result[5][0])
    assert all(np.isnan(result[i][0]) for i in (0, 1, 2, 4))


@pytest.mark.parametrize('functions', [(None, helix_cap_intersection), (helix_cylinder_intersection, None)])
def test_noncallables_reject(functions):
    with pytest.raises(ValueError, match='callable'):
        intersector_class(*functions)

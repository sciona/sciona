"""Verify tangent adapter broadcasting, regulation, and output contracts."""
import numpy as np
import pytest

from sciona.decomposed_trackml_tangent_lifecycle import construction, tangent
from sciona.atoms.physics.helix_tangent import construct_helix_from_tangent, oriented_helix_tangent
from sciona.trackml_nearest import helix_from_tangent, unit_tangent


@pytest.mark.parametrize('size',[0,1,7])
@pytest.mark.parametrize('pitch',[2.,np.float64(-3.)])
def test_scalar_pitch_and_recipe_controls(size,pitch):
    values=[np.full(size,v,dtype=np.float64) for v in (1,2,3,.5,-.4,0)]
    controls=dict(min_uz0=.02,max_mg_over_qB=4.)
    actual=construction(*values,pitch,operation=construct_helix_from_tangent,**controls)
    expected=helix_from_tangent(*values,pitch,**controls)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('size',[0,1,7])
def test_oriented_tangent_adapter(size):
    values=[np.full(size,v,dtype=np.float64) for v in (1,2,3,-1,0,0,2,4)]
    np.testing.assert_array_equal(tangent(*values,operation=oriented_helix_tangent),unit_tangent(*values))


@pytest.mark.parametrize('center,radius',[
    (np.zeros((1,2),dtype=np.float32),np.ones(1)),
    (np.zeros((2,2)),np.ones(1)),
    (np.zeros((1,2)),np.array([-1.])),
    (np.full((1,2),np.nan),np.ones(1)),
    (np.zeros((1,2)),np.ones((1,1))),
])
def test_bad_construction_output(center,radius):
    with pytest.raises(ValueError):
        construction(*[np.ones(1) for _ in range(6)],2.,operation=lambda *a:(center,radius))


@pytest.mark.parametrize('result',[np.ones((1,3),dtype=np.float32),np.ones((3,1)),np.full((1,3),np.inf)])
def test_bad_tangent_output(result):
    with pytest.raises(ValueError):
        tangent(*[np.ones(1) for _ in range(8)],operation=lambda *a:result)


def test_invalid_pitch_rejected_before_provider():
    def forbidden(*args):raise AssertionError('Invalid pitch reached provider')
    with pytest.raises(ValueError):
        construction(*[np.ones(2) for _ in range(6)],np.ones(3),operation=forbidden)

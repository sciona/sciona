"""Independent analytical contracts and explicit witness limits for intersections."""
import numpy as np
import pytest
from sciona.ghost.abstract import AbstractArray
from sciona.atoms.physics.helix_cylinder import helix_cylinder_intersection as cylinder, concrete_only
from sciona.atoms.particle_tracking.track_matching.atoms import helix_cap_intersection as plane
from sciona.atoms.particle_tracking.track_matching.witnesses import witness_helix_cap_intersection


@pytest.mark.parametrize('scale', [.01, 1., 100.])
@pytest.mark.parametrize('pitch_sign', [-1., 1.])
@pytest.mark.parametrize('phase', [-np.pi/2, 0., np.pi/2])
def test_plane_analytical_quarter_turn(scale, pitch_sign, phase):
    x,y,z,xm,ym,r,p,target=[np.array([v]) for v in
        (3.*scale, 2.*scale, scale, scale, 2.*scale, 2.*scale,
         pitch_sign*2*np.pi*scale, scale+phase*pitch_sign*scale)]
    result=plane(x,y,z,xm,ym,r,p,target)
    expected=(xm+2*scale*np.cos(phase),ym+2*scale*np.sin(phase),target,np.array([phase]))
    for a,b in zip(result,expected,strict=True):
        np.testing.assert_allclose(a,b,rtol=2e-14,atol=2e-14*scale)
    np.testing.assert_array_equal(target,expected[2])


def test_plane_radius_is_unused_and_inputs_unchanged():
    args=[np.array([v]) for v in (3.,2.,0.,1.,2.,2.,3.,1.)]
    saved=[v.copy() for v in args]
    first=plane(*args)
    args[5]=np.array([99.])
    second=plane(*args)
    for a,b in zip(first,second,strict=True):np.testing.assert_array_equal(a,b)
    for i in (0,1,2,3,4,6,7):np.testing.assert_array_equal(args[i],saved[i])


def test_plane_witness_only_describes_output_shape():
    a=AbstractArray(shape=(7,),dtype='float64')
    outputs=witness_helix_cap_intersection(*([a]*8))
    assert len(outputs)==4 and all(o.shape==(7,) and o.dtype=='float64' for o in outputs)


def test_cylinder_abstract_witness_rejects():
    a=AbstractArray(shape=(7,),dtype='float64')
    with pytest.raises(ValueError,match='Concrete'):
        concrete_only(*([a]*10))


def test_cylinder_direction_sign_does_not_underflow():
    args=[np.array([v]) for v in (3.,0.,0.,2.,0.,1.,1e-200,1e-200,5.,0.)]
    output=cylinder(*args)
    assert output[4][0] and output[3][0]>0
    np.testing.assert_allclose(output[0]**2+output[1]**2,5.)


def test_cylinder_overflow_rejects():
    args=[np.array([v]) for v in (3.,0.,0.,1e308,0.,1.,1.,1.,5.,0.)]
    with pytest.raises(ValueError):cylinder(*args)

"""Synthetic source parity and independent helix differential geometry."""
import numpy as np
import pytest
from sciona.atoms.physics.helix_tangent import construct_helix_from_tangent, oriented_helix_tangent, concrete_only
from sciona.trackml_nearest import _from_tangent, _unit_tangent


@pytest.mark.parametrize('size',[1,7,31])
@pytest.mark.parametrize('minimum',[.001,.4])
@pytest.mark.parametrize('maximum',[.25,1e6])
def test_construction_exact_source_and_geometry(size,minimum,maximum):
    rng=np.random.default_rng(19)
    p=rng.normal(size=(size,3));t=rng.normal(size=(size,3));h=rng.uniform(-5.,5.,size)
    t[0,2]=0.
    before=[v.copy() for v in (p,t,h)]
    expected=_from_tangent(*p.T,*t.T,h,min_uz0=minimum,max_mg_over_qB=maximum)
    center,radius=construct_helix_from_tangent(p,t,h,minimum,maximum)
    np.testing.assert_array_equal(center,np.column_stack(expected[:2]))
    np.testing.assert_array_equal(radius,expected[2])
    np.testing.assert_allclose(np.linalg.norm(p[:,:2]-center,axis=1),radius)
    for a,b in zip((p,t,h),before):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('sign',[-1.,1.])
@pytest.mark.parametrize('pitch_sign',[-1.,1.])
@pytest.mark.parametrize('size',[1,7,31])
def test_tangent_exact_source_and_analytic_derivative(sign,pitch_sign,size):
    rng=np.random.default_rng(21)
    center=rng.normal(size=(size,2));radius=rng.uniform(.2,4.,size)
    phase=rng.uniform(-3.,3.,size);pitch=pitch_sign*rng.uniform(.2,5.,size)
    points=np.column_stack((center[:,0]+radius*np.cos(phase),center[:,1]+radius*np.sin(phase),rng.normal(size=size)))
    direction=np.full(size,sign)
    result=oriented_helix_tangent(points,center,radius,pitch,direction)
    expected=_unit_tangent(*points.T,direction,*center.T,radius,pitch)
    np.testing.assert_array_equal(result,expected)
    derivative=np.column_stack((-radius*np.sin(phase),radius*np.cos(phase),pitch/(2*np.pi)))
    expected=derivative/np.linalg.norm(derivative,axis=1)[:,None]*(sign*pitch_sign)
    np.testing.assert_allclose(result,expected,atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(result,axis=1),1.)
    np.testing.assert_array_equal(oriented_helix_tangent(points,center,radius,pitch,-direction),-result)


@pytest.mark.parametrize('control',[0.,-1.,np.inf,np.nan,True])
def test_bad_controls(control):
    with pytest.raises(ValueError):construct_helix_from_tangent(np.zeros((1,3)),np.ones((1,3)),np.ones(1),control,1.)
    with pytest.raises(ValueError):construct_helix_from_tangent(np.zeros((1,3)),np.ones((1,3)),np.ones(1),1.,control)


@pytest.mark.parametrize('bad',[np.zeros((1,2)),np.zeros((1,3),dtype=np.float32),np.full((1,3),np.nan)])
def test_bad_points(bad):
    with pytest.raises(ValueError):construct_helix_from_tangent(bad,np.ones((1,3)),np.ones(1),.1,1.)
    with pytest.raises(ValueError):oriented_helix_tangent(bad,np.zeros((1,2)),np.ones(1),np.ones(1),np.ones(1))


def test_empty_and_singular_and_overflow():
    c,r=construct_helix_from_tangent(np.empty((0,3)),np.empty((0,3)),np.empty(0),.1,1.)
    assert c.shape==(0,2) and r.shape==(0,)
    assert oriented_helix_tangent(np.empty((0,3)),c,r,r,r).shape==(0,3)
    for radius,pitch,direction in [(-1.,1.,1.),(1.,0.,1.),(1.,1.,0.),(1.,1e308,1e308)]:
        with pytest.raises(ValueError):oriented_helix_tangent(np.ones((1,3)),np.zeros((1,2)),np.array([radius]),np.array([pitch]),np.array([direction]))
    with pytest.raises(ValueError):construct_helix_from_tangent(np.zeros((1,3)),np.array([[1.,1.,1e-308]]),np.array([1e308]),1e-308,1.)
    with pytest.raises(ValueError):concrete_only()

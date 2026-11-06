"""Correction adapter domain, state mutation and invocation isolation contracts."""
import numpy as np
import pytest
from sciona.decomposed_trackml_corrections import corrector_class
from sciona.trackml_corrections import HelixCorrector
from sciona.atoms.particle_tracking.detector_corrections.atoms import perturbative_cap_correction, perturbative_cylinder_correction
from scripts.review_trackml_correction_reuse import context, source_class


def setup(points=None):
    x,y,z=np.array([[2.,3.],[1.,2.],[0.,1.]]) if points is None else points
    radius=np.full(2,2.);pitch=np.full(2,3.);direction=np.ones(2)
    state=context(x,y,z,radius,pitch,direction,np.array([np.nan,.2]),np.array([.1,np.nan]))
    return state,[x.copy(),y.copy(),z.copy()],np.zeros(2,dtype=np.int8)


@pytest.mark.parametrize('name', ['cap','cylinder'])
def test_exact_source_missing_coefficients_and_in_place_update(name):
    cls=corrector_class(HelixCorrector,perturbative_cap_correction,perturbative_cylinder_correction)
    state,args,layer=setup()
    source,_=source_class()
    method='correctCapIntersections' if name=='cap' else 'correctCylinderIntersections'
    kwargs={} if name=='cap' else {'mask':np.ones(2,dtype=bool)}
    expected=getattr(source,method)(state,*[a.copy() for a in args],layer,**kwargs)
    actual=getattr(cls,method)(state,*args,layer,**kwargs)
    for i,(a,b) in enumerate(zip(actual,expected,strict=True)):
        assert a is args[i]
        np.testing.assert_array_equal(a,b)


def test_invalid_masked_rows_do_not_reach_provider():
    seen=[]
    def cylinder(*args):
        seen.append(len(args[0]))
        return perturbative_cylinder_correction(*args)
    cls=corrector_class(HelixCorrector,perturbative_cap_correction,cylinder)
    state,args,layer=setup()
    for a in args:a[1]=np.nan
    result=cls.correctCylinderIntersections(state,*args,layer,mask=np.array([True,False]))
    assert seen==[1] and all(np.isnan(a[1]) for a in result)


@pytest.mark.parametrize('mode', ['cap_axis','cap_pitch','cylinder_axis','cylinder_radius'])
def test_undefined_domain_rejects(mode):
    cls=corrector_class(HelixCorrector,perturbative_cap_correction,perturbative_cylinder_correction)
    state,args,layer=setup()
    if mode.endswith('axis'):args[0][:]=args[1][:]=0.
    else:
        params=list(state.helixParams())
        params[4 if mode=='cap_pitch' else 3][:]=0.
    with pytest.raises(ValueError):
        if mode.startswith('cap'):cls.correctCapIntersections(state,*args,layer)
        else:cls.correctCylinderIntersections(state,*args,layer,mask=np.ones(2,dtype=bool))


def test_clone_subset_bindings_and_no_map_shortcut():
    def forbidden(*args):raise AssertionError('No maps')
    cls=corrector_class(HelixCorrector,forbidden,forbidden)
    other=corrector_class(HelixCorrector,perturbative_cap_correction,perturbative_cylinder_correction)
    assert cls.clone.__globals__['HelixCorrector'] is cls
    assert cls.subset.__globals__['HelixCorrector'] is cls
    assert other.clone.__globals__['HelixCorrector'] is other
    assert HelixCorrector.clone.__globals__['HelixCorrector'] is HelixCorrector
    state,args,layer=setup();state._run.layer_functions=None
    for name in ('correctCapIntersections','correctCylinderIntersections'):
        result=getattr(cls,name)(state,*args,layer,**({'mask':np.ones(2,dtype=bool)} if 'Cylinder' in name else {}))
        assert all(a is b for a,b in zip(result,args))

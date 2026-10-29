"""Synthetic exact-source feature parity and physical/unit boundary checks."""
import numpy as np
import pytest
from sciona.flavours_features import prepare_features,classifier_views
from sciona.atoms.ml.flavours_feature_views import (
    positive_longitudinal_kinematics,assemble_source_feature_views,enrich_corrected_mass_views,
    witness_positive_longitudinal_kinematics,
)
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature


def physical(rows=23,daughters=3):
    rng=np.random.default_rng(943)
    p=rng.uniform(.5,8.,(rows,daughters));pt=p*rng.uniform(0.,1.,p.shape)
    p[0,0]=pt[0,0]=0.
    return p,pt,rng.uniform(.2,3.,rows),rng.uniform(1.,5.,rows),rng.uniform(.1,2.,rows)


@pytest.mark.parametrize('excluded',[0,17,45])
def test_feature_assembly_and_enrichment_match_retained_reference(excluded):
    base=np.random.default_rng(522).normal(size=(23,46));inputs=physical();before=[v.copy() for v in (base,*inputs)]
    quantities=positive_longitudinal_kinematics(*inputs)
    observed=assemble_source_feature_views(base,*quantities,excluded)
    reference=prepare_features(base,*inputs,excluded)
    for key in reference:np.testing.assert_array_equal(observed[key],reference[key])
    mass=quantities[0]*1.13+.7
    actual_views=enrich_corrected_mass_views(observed,mass)
    expected_views=classifier_views(reference,mass)
    for key in expected_views:np.testing.assert_array_equal(actual_views[key],expected_views[key])
    for actual,old in zip((base,*inputs),before):np.testing.assert_array_equal(actual,old)


@pytest.mark.parametrize('daughters',[1,4,7])
def test_generic_physical_operation_supports_other_daughter_counts_and_unit_scaling(daughters):
    inputs=physical(daughters=daughters);p,pt,parent,distance,time=inputs
    proxy,total,speed,longitudinal=positive_longitudinal_kinematics(*inputs)
    np.testing.assert_allclose(longitudinal,np.sqrt(p*p-pt*pt).sum(axis=1),rtol=1e-14)
    np.testing.assert_allclose(proxy,np.hypot(parent,longitudinal)*time/distance,rtol=1e-14)
    scaled=positive_longitudinal_kinematics(p*2,pt*2,parent*2,distance*3,time*4)
    for actual,expected in zip(scaled,(proxy*8/3,total*2,speed*3/4,longitudinal*2)):
        np.testing.assert_allclose(actual,expected,rtol=1e-14)


@pytest.mark.parametrize('fault',['negative_momentum','excess_transverse','zero_distance','zero_lifetime','nonfinite','misaligned'])
def test_invalid_physical_inputs_fail(fault):
    values=list(physical())
    if fault=='negative_momentum':values[0][0,0]=-1
    elif fault=='excess_transverse':values[1][0,0]=1
    elif fault=='zero_distance':values[3][0]=0
    elif fault=='zero_lifetime':values[4][0]=0
    elif fault=='nonfinite':values[2][0]=np.inf
    else:values[2]=values[2][:-1]
    with pytest.raises(ValueError):positive_longitudinal_kinematics(*values)


def test_physical_witness_tracks_mass_momentum_and_velocity_units():
    momentum=DimensionalSignature(M=1,L=1,T=-1)
    p=AbstractArray(shape=(23,3),dim=momentum)
    parent=AbstractArray(shape=(23,),dim=momentum)
    distance=AbstractArray(shape=(23,),dim=DimensionalSignature(L=1))
    time=AbstractArray(shape=(23,),dim=DimensionalSignature(T=1))
    proxy,total,speed,longitudinal=witness_positive_longitudinal_kinematics(p,p,parent,distance,time)
    assert proxy.dim==DimensionalSignature(M=1) and total.dim==longitudinal.dim==momentum
    assert speed.dim==DimensionalSignature(L=1,T=-1)
    with pytest.raises(ValueError,match='units'):witness_positive_longitudinal_kinematics(p,p,parent,time,distance)


def test_source_view_width_exclusion_and_singular_ratio_fail():
    base=np.ones((23,46));quantities=positive_longitudinal_kinematics(*physical())
    for excluded in [-1,46,True,1.5]:
        with pytest.raises(ValueError):assemble_source_feature_views(base,*quantities,excluded)
    with pytest.raises(ValueError):assemble_source_feature_views(base[:,:45],*quantities,0)
    prepared=assemble_source_feature_views(base,*quantities,0)
    with pytest.raises((ValueError,FloatingPointError)):enrich_corrected_mass_views(prepared,np.full(23,-1e-10))

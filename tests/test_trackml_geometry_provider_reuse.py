"""Analytic synthetic geometry independent of retained source formulas."""
import numpy as np
import pytest
from sciona.atoms.particle_tracking.helix_geometry import atoms


@pytest.mark.parametrize('seed',range(4))
@pytest.mark.parametrize('scale',[.01,1.,100.])
def test_analytic_circle_and_pitch_across_units(seed,scale):
    rng=np.random.RandomState(seed)
    n=32;center=rng.uniform(-3,3,(2,n))*scale;radius=rng.uniform(.5,4,n)*scale
    phase=rng.uniform(-np.pi,np.pi,n);step=rng.uniform(.2,.8,n)*rng.choice([-1.,1.],n)
    pitch=rng.uniform(.2,7,n)*rng.choice([-1.,1.],n)*scale;z=rng.normal(size=n)*scale
    points=[(center[0]+radius*np.cos(phase+j*step),center[1]+radius*np.sin(phase+j*step),z+j*step*pitch/(2*np.pi)) for j in range(3)]
    values=[points[j][k] for j in range(3) for k in (0,1)]
    before=[v.copy() for v in values]
    actual=atoms.circle_from_three_points(*values)
    for a,b in zip(actual,(*center,radius)):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10*scale)
    two=atoms.helix_pitch_from_two_points(*points[0],*points[1],*center)
    three=atoms.helix_pitch_least_squares(*points[0],*points[1],*points[2],*center)
    for result in (two,three):
        np.testing.assert_allclose(result[0],pitch,rtol=1e-11,atol=1e-11*scale)
        np.testing.assert_allclose(result[1],step,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(result[2],step*pitch/(2*np.pi),rtol=1e-11,atol=1e-11*scale)
    assert np.max(three[3])<1e-24*scale**2
    for a,b in zip(values,before):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('replacement',[0.,.001,-.02])
def test_zero_pitch_policy_is_explicit(replacement):
    phase=np.array([.1,.6,1.1]);points=[np.array([v]) for v in phase]
    xyz=[(np.cos(p),np.sin(p),np.zeros(1)) for p in points]
    for result in (atoms.helix_pitch_from_two_points(*xyz[0],*xyz[1],np.zeros(1),np.zeros(1),zero_pitch=replacement),
                   atoms.helix_pitch_least_squares(*xyz[0],*xyz[1],*xyz[2],np.zeros(1),np.zeros(1),zero_pitch=replacement)):
        np.testing.assert_array_equal(result[0],np.array([replacement]))
        np.testing.assert_allclose(result[2],replacement*result[1]/(2*np.pi),rtol=1e-15,atol=0)

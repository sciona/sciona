"""Generic synthetic geometry preserves source semantics through existing providers."""
import numpy as np
import pytest
from sciona.atoms.particle_tracking.helix_geometry import atoms
from sciona import decomposed_trackml_geometry as adapter
from sciona.trackml_geometry import circle_from_three_points
from sciona.trackml_helix import pitch_from_two_points,pitch_least_squares

CASES=[(adapter.circle,circle_from_three_points,atoms.circle_from_three_points,6,1e6,'large_radius',3),
       (adapter.pitch_two,pitch_from_two_points,atoms.helix_pitch_from_two_points,8,.001,'zero_pitch',3),
       (adapter.pitch_three,pitch_least_squares,atoms.helix_pitch_least_squares,11,.001,'zero_pitch',4)]

@pytest.mark.parametrize('case',CASES)
def test_empty_batches_bypass_provider(case):
    fn,reference,operation,n,option,key,outputs=case
    def forbidden(*args,**kwargs):raise AssertionError('Empty batch reached nonempty provider')
    result=fn(*[np.empty(0) for _ in range(n)],option,forbidden)
    assert len(result)==outputs
    assert all(v.shape==(0,) and v.dtype==np.float64 for v in result)

@pytest.mark.parametrize('seed',range(6))
def test_generic_coordinates_match(seed):
    rng=np.random.RandomState(seed)
    center=rng.normal(size=(2,24));radius=rng.uniform(.5,12,24)
    phase=rng.uniform(-np.pi,np.pi,24);step=rng.uniform(.1,.7,24)*rng.choice([-1,1],24)
    pitch=rng.uniform(-8,8,24);z=rng.normal(size=24)
    points=[(center[0]+radius*np.cos(phase+j*step),center[1]+radius*np.sin(phase+j*step),z+j*step*pitch/(2*np.pi)) for j in range(3)]
    inputs=[[points[j][k] for j in range(3) for k in (0,1)],
            [*points[0],*points[1],*center],[*points[0],*points[1],*points[2],*center]]
    for case,values in zip(CASES,inputs):
        fn,reference,operation,n,option,key,outputs=case;saved=[v.copy() for v in values];calls=[]
        def invoke(*args,**kwargs):
            calls.append(True)
            return operation(*args,**kwargs)
        actual=fn(*values,option,invoke);expected=reference(*values,**{key:option})
        assert calls==[True]
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        for a,b in zip(values,saved):np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('case',[CASES[0],CASES[2]])
def test_source_undefined_geometry_rejects_before_substitution(case):
    fn,reference,operation,n,option,key,outputs=case
    def forbidden(*args,**kwargs):raise AssertionError('Undefined geometry reached provider')
    with pytest.raises(ValueError):fn(*[np.zeros(1) for _ in range(n)],option,forbidden)

@pytest.mark.parametrize('case',CASES)
def test_misaligned_vectors_reject(case):
    fn,reference,operation,n,option,key,outputs=case
    values=[np.ones(2) for _ in range(n)];values[-1]=np.ones(3)
    with pytest.raises(ValueError):fn(*values,option,operation)

@pytest.mark.parametrize('magnitude',[0.,1e-200])
def test_underflow_tangent_rejects(magnitude):
    with pytest.raises(ValueError):
        adapter.circle(*[np.full(1,magnitude) for _ in range(6)],1e6,atoms.circle_from_three_points)


def test_collinear_and_repeated_nonzero_points():
    values=[np.array(v,dtype=np.float64) for v in ([1,2,1],[1,3,2],[2,2,1],[2,3,2],[3,2,3],[3,3,4])]
    actual=adapter.circle(*values,1e6,atoms.circle_from_three_points)
    expected=circle_from_three_points(*values)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_binding_is_invocation_local():
    from sciona.decomposed_trackml_lifecycle import lifecycle_class
    from sciona.trackml_fitting import FittedTrackExtension
    from sciona.trackml_geometry import _source_circle
    args=(atoms.circle_from_three_points,atoms.helix_pitch_from_two_points,atoms.helix_pitch_least_squares)
    first=lifecycle_class(*args);second=lifecycle_class(*args)
    assert first.fitTracks.__code__ is FittedTrackExtension.fitTracks.__code__
    assert first.fitTracks.__globals__ is not second.fitTracks.__globals__
    assert FittedTrackExtension.fitTracks.__globals__['circleFromThreePoints'] is _source_circle
    assert first.fitTracks.__globals__['circleFromThreePoints'] is not _source_circle


@pytest.mark.parametrize('missing',range(3))
def test_binding_rejects_noncallable(missing):
    functions=[atoms.circle_from_three_points,atoms.helix_pitch_from_two_points,atoms.helix_pitch_least_squares]
    functions[missing]=None
    with pytest.raises(ValueError):adapter.bind(*functions)

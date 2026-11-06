"""Synthetic source parity and coordinate invariants for existing helix direction."""
import numpy as np
import pytest
from sciona.atoms.particle_tracking.helix_geometry.atoms import helix_direction_from_two_points
from sciona.trackml_nearest import _direction


@pytest.mark.parametrize('size', [1, 5, 31])
@pytest.mark.parametrize('offset', [0., 17.])
@pytest.mark.parametrize('pitch_sign', [-1., 1.])
@pytest.mark.parametrize('mode', ['axial', 'planar', 'coincident'])
def test_source_parity_and_coordinate_invariants(size, offset, pitch_sign, mode):
    rng = np.random.default_rng(713)
    xm = rng.uniform(-2., 2., size)+offset
    ym = rng.uniform(-2., 2., size)-offset
    pitch = pitch_sign*rng.uniform(.2, 4., size)
    radius = rng.uniform(.5, 3., size)
    phase = rng.uniform(.1, 1., size)
    x1, y1, z1 = xm+radius, ym.copy(), rng.uniform(-1., 1., size)
    x2, y2 = xm+radius*np.cos(phase), ym+radius*np.sin(phase)
    z2 = z1+rng.uniform(.2, 1., size) if mode == 'axial' else z1.copy()
    if mode == 'coincident':
        x2, y2 = x1.copy(), y1.copy()
    args = [xm, ym, pitch, x1, y1, z1, x2, y2, z2]
    before = [v.copy() for v in args]
    result = helix_direction_from_two_points(*args)
    np.testing.assert_array_equal(result, _direction(*args))
    expected = z2-z1 if mode == 'axial' else pitch if mode == 'planar' else np.zeros(size)
    np.testing.assert_array_equal(result, expected)
    reverse = helix_direction_from_two_points(xm,ym,pitch,x2,y2,z2,x1,y1,z1)
    np.testing.assert_array_equal(reverse, -result)
    # Common positive length-unit change preserves direction and scales the indicator.
    np.testing.assert_allclose(helix_direction_from_two_points(*[v*8. for v in args]), result*8., atol=1e-13)
    translated = [v.copy() for v in args]
    for index in (0,3,6): translated[index] += 32.
    for index in (1,4,7): translated[index] -= 16.
    for index in (5,8): translated[index] += 8.
    np.testing.assert_allclose(helix_direction_from_two_points(*translated), result, atol=1e-13)
    for original, value in zip(before,args):
        np.testing.assert_array_equal(value,original)
    assert not any(np.shares_memory(result,v) for v in args)


def test_empty_batch_is_outside_existing_provider_contract():
    import icontract
    with pytest.raises(icontract.ViolationError):
        helix_direction_from_two_points(*[np.empty(0) for _ in range(9)])

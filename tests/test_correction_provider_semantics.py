"""Synthetic analytical checks for reusable local-frame displacements."""
import numpy as np
import pytest
from sciona.ghost.abstract import AbstractArray
from sciona.atoms.particle_tracking.detector_corrections.atoms import (
    perturbative_cap_correction as cap, perturbative_cylinder_correction as cylinder,
)
from sciona.atoms.particle_tracking.detector_corrections.witnesses import (
    witness_perturbative_cap_correction, witness_perturbative_cylinder_correction,
)


@pytest.mark.parametrize('kind', ['cap', 'cylinder'])
@pytest.mark.parametrize('scale', [.01, 1., 100.])
@pytest.mark.parametrize('orientation', [-1., 0., 1.])
@pytest.mark.parametrize('pitch_sign', [-1., 1.])
def test_local_basis_displacements_and_unit_scaling(kind, scale, orientation, pitch_sign):
    # Four axis directions make the radial and tangent bases analytical.
    radial = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    tangent = np.array([[0., 1.], [-1., 0.], [0., -1.], [1., 0.]])
    x, y = (3*scale*radial).T.copy()
    z = np.full(4, 4*scale)
    pitch = np.full(4, pitch_sign*2*np.pi*scale)
    sign = np.full(4, orientation)
    first, second = np.full(4, .2*scale**2), np.full(4, .3*scale**2)
    if kind == 'cap':
        args = [x, y, pitch, sign, first, second]
        expected = (3*scale*radial + orientation*scale*(.2*radial+.3*tangent)).T
        fn = cap
    else:
        # Radius = pitch-per-radian = scale gives curvature radius 2*scale.
        args = [x, y, z, np.full(4, scale), pitch, sign, first, second]
        xy = 3*scale*radial + orientation*.15*scale*tangent
        expected = [xy[:, 0], xy[:, 1], z+orientation*.1*scale]
        fn = cylinder
    saved = [a.copy() for a in args]
    output = fn(*args)
    for actual, wanted in zip(output, expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=2e-14, atol=2e-14*scale)
        assert actual.dtype == np.float64 and actual.shape == (4,)
        assert all(not np.shares_memory(actual, a) for a in args)
    for actual, wanted in zip(args, saved, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_cylinder_zero_pitch_is_defined():
    args = [np.array([v]) for v in (3., 0., 4., 2., 0., 1., .4, .6)]
    for actual, expected in zip(cylinder(*args), (3., .3, 4.2), strict=True):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('witness,arity,outputs', [
    (witness_perturbative_cap_correction, 6, 2),
    (witness_perturbative_cylinder_correction, 8, 3),
])
def test_witness_only_describes_shapes(witness, arity, outputs):
    a = AbstractArray(shape=(7,), dtype='float64')
    result = witness(*([a]*arity))
    assert len(result) == outputs
    assert all(v.shape == (7,) and v.dtype == 'float64' for v in result)

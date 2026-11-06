"""Synthetic unit-independent checks of normalized residual decisions."""
import numpy as np
import pytest

from sciona.atoms.particle_tracking.track_matching.atoms import bayesian_neighbor_evaluation as decision
from sciona.atoms.particle_tracking.track_matching.witnesses import witness_bayesian_neighbor_evaluation
from sciona.ghost.abstract import AbstractArray


@pytest.mark.parametrize('first_scale,second_scale', [(1., 1.), (.01, 100.), (100., .01)])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('trust', [.5, 1., 2.])
def test_independent_component_units_and_strict_flags(first_scale, second_scale, sign, trust):
    # Background/error=sqrt(pi/2) in each component gives cut=sqrt(2*log(4)).
    # Distances are deliberately away from boundaries for unit conversion tests.
    cut = np.sqrt(2*np.log(4.))
    ratios = np.array([0., .25, .75, 1.5, 3.])
    error = np.ones(5)
    background = np.full(5, np.sqrt(np.pi/2))
    args = [sign*ratios*cut*first_scale, np.zeros(5),
            background*first_scale, background*second_scale,
            error*first_scale, error*second_scale]
    saved = [a.copy() for a in args]
    good, dubious = decision(*args, cut_factor=1., dist_trust=trust)
    np.testing.assert_array_equal(good, ratios < 1.)
    np.testing.assert_array_equal(dubious, ratios > trust)
    assert good.dtype == dubious.dtype == np.bool_
    assert good.shape == dubious.shape == (5,)
    for actual, expected in zip(args, saved, strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(good, actual) and not np.shares_memory(dubious, actual)


def test_zero_background_and_independent_flags():
    one, zero = np.ones(3), np.zeros(3)
    good, dubious = decision(np.array([0., 1., -1.]), zero, zero, zero, one, one)
    np.testing.assert_array_equal(good, [False, False, False])
    np.testing.assert_array_equal(dubious, [False, True, True])
    good, dubious = decision(one, zero, one, one, one, one, cut_factor=1., dist_trust=.1)
    assert good.all() and dubious.all()


def test_witness_only_describes_boolean_shapes():
    a = AbstractArray(shape=(7,), dtype='float64')
    result = witness_bayesian_neighbor_evaluation(*([a]*6), cut_factor=1., dist_trust=.8)
    assert len(result) == 2
    assert all(v.shape == (7,) and v.dtype == 'bool' for v in result)

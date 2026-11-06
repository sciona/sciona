"""Independent cut-boundary checks and exact parity with the source recipe."""
import numpy as np
import pytest

from sciona.aptos_ensemble import CUT_POINTS, classify
from sciona.atoms.numerical.statistics.cut_indices import ordered_cut_indices, witness_ordered_cut_indices


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('policy', ['upper', 'lower'])
def test_source_cut_neighbors_match_exactly(dtype, policy):
    cuts = np.asarray(CUT_POINTS, dtype=np.float64)
    source_values = np.asarray(CUT_POINTS, dtype=dtype)
    values = np.concatenate([[-100.], np.nextafter(source_values, dtype(-np.inf)),
        source_values, np.nextafter(source_values, dtype(np.inf)), [100.]]).astype(dtype)
    np.testing.assert_array_equal(ordered_cut_indices(values, cuts, policy), classify(values, tie_policy=policy))


@pytest.mark.parametrize('policy,expected', [
    ('upper', [0, 1, 1, 2, 2, 3, 3]), ('lower', [0, 0, 1, 1, 2, 2, 3]),
])
def test_arbitrary_signed_cut_points(policy, expected):
    values = np.array([-11., -10., -9., 0., 1., 7., 8.])
    cuts = np.array([-10., 0., 7.])
    np.testing.assert_array_equal(ordered_cut_indices(values, cuts, policy), expected)


@pytest.mark.parametrize('policy', ['upper', 'lower'])
def test_empty_cut_vector_has_one_bin(policy):
    np.testing.assert_array_equal(ordered_cut_indices(np.array([-3., 5.]), np.array([]), policy), [0, 0])


@pytest.mark.parametrize('policy,expected', [('upper', [1, 1]), ('lower', [0, 0])])
def test_signed_zero_equality(policy, expected):
    np.testing.assert_array_equal(ordered_cut_indices(np.array([-0., 0.]), np.array([0.]), policy), expected)


def test_extreme_finite_cuts_do_not_overflow_validation():
    limit = np.finfo(np.float64).max
    np.testing.assert_array_equal(ordered_cut_indices(np.array([-limit, 0., limit]),
        np.array([-limit, limit]), 'upper'), [1, 1, 2])


def test_noncontiguous_readonly_inputs_and_fresh_output():
    values = np.arange(10., dtype=np.float64)[::2]
    cuts = np.array([2., 6.])
    before_values, before_cuts = values.copy(), cuts.copy()
    values.setflags(write=False)
    cuts.setflags(write=False)
    result = ordered_cut_indices(values, cuts, 'upper')
    np.testing.assert_array_equal(result, [0, 1, 1, 2, 2])
    np.testing.assert_array_equal(values, before_values)
    np.testing.assert_array_equal(cuts, before_cuts)
    assert result.dtype.kind in 'iu'
    assert not np.shares_memory(result, values) and not np.shares_memory(result, cuts)


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.array([np.inf]), np.array([-np.inf]),
    np.array([1], dtype=np.int64), np.array([True]), np.array([1.], dtype=np.float16),
    np.array([1.+0j]), np.array([1.], dtype=object), np.array([[1.]]), [1.]])
@pytest.mark.parametrize('argument', ['values', 'cut_points'])
def test_invalid_numeric_inputs_reject(bad, argument):
    arguments = dict(values=np.array([0.]), cut_points=np.array([1.]), tie_policy='upper')
    arguments[argument] = bad
    with pytest.raises(ValueError):
        ordered_cut_indices(**arguments)


@pytest.mark.parametrize('cuts', [np.array([1., 1.]), np.array([2., 1.]), np.array([-0., 0.])])
def test_non_strict_cuts_reject(cuts):
    with pytest.raises(ValueError):
        ordered_cut_indices(np.array([0.]), cuts, 'upper')


@pytest.mark.parametrize('policy', ['nearest', '', None, True])
def test_implicit_or_unknown_tie_policy_rejects(policy):
    with pytest.raises(ValueError):
        ordered_cut_indices(np.array([0.]), np.array([1.]), policy)


def test_empty_values_reject():
    with pytest.raises(ValueError):
        ordered_cut_indices(np.array([]), np.array([1.]), 'upper')


def test_symbolic_execution_is_not_claimed():
    with pytest.raises(ValueError):
        witness_ordered_cut_indices(None, None, 'upper')

"""Synthetic analytical checks for domain-independent ordered selection."""
import numpy as np
import pytest
from sciona.atoms.particle_tracking.track_matching.atoms import greedy_track_commit as select
from sciona.atoms.particle_tracking.track_matching.witnesses import witness_greedy_track_commit
from sciona.ghost.abstract import AbstractArray


@pytest.mark.parametrize('fraction,accepted', [(.5, True), (np.nextafter(.5, 0.), False), (np.nextafter(.5, 1.), True)])
@pytest.mark.parametrize('loss_limit', [1, 2])
def test_loss_fraction_inclusive_boundary(fraction, accepted, loss_limit):
    matrix = np.array([[1, 2, 3, 4]], dtype=np.int64)
    used = np.array([False, True, True, False, False])
    before = used.copy()
    actual, claimed = select(matrix, used, min_nhits=2, max_nloss=loss_limit,
                             max_loss_fraction=fraction)
    accept = accepted and loss_limit == 2
    np.testing.assert_array_equal(actual, [[0, 0, 3, 4]] if accept else [[0, 0, 0, 0]])
    np.testing.assert_array_equal(claimed, [True, True, True, accept, accept])
    np.testing.assert_array_equal(matrix, [[1, 2, 3, 4]])
    np.testing.assert_array_equal(used, before)
    assert not np.shares_memory(actual, matrix) and not np.shares_memory(claimed, used)


@pytest.mark.parametrize('limit', [0, 1, 2, 3])
def test_count_limit_and_repeated_positions(limit):
    matrix = np.array([[1, 1, 0], [1, 2, 3], [4, 5, 0]], dtype=np.int64)
    actual, claimed = select(matrix, np.zeros(6, dtype=bool), min_nhits=2, max_nrows=limit)
    expected = np.array([[1, 1, 0], [0, 2, 3], [4, 5, 0]])
    expected[limit:] = 0
    np.testing.assert_array_equal(actual, expected)
    mask = np.zeros(6, dtype=bool)
    mask[expected] = True
    np.testing.assert_array_equal(claimed, mask)


@pytest.mark.parametrize('shape', [(0, 0), (0, 3), (3, 0), (3, 4)])
def test_empty_and_all_zero_rows(shape):
    matrix = np.zeros(shape, dtype=np.int64)
    actual, claimed = select(matrix, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(actual, matrix)
    np.testing.assert_array_equal(claimed, [True, False])
    assert actual.dtype == np.int64 and claimed.dtype == np.bool_


def test_order_dependence_and_resource_renaming():
    matrix = np.array([[1, 2], [1, 3]], dtype=np.int64)
    actual, _ = select(matrix, np.zeros(4, dtype=bool), min_nhits=2)
    reversed_selection, _ = select(matrix[::-1], np.zeros(4, dtype=bool), min_nhits=2)
    np.testing.assert_array_equal(actual, [[1, 2], [0, 0]])
    np.testing.assert_array_equal(reversed_selection, [[1, 3], [0, 0]])
    rename = np.array([0, 3, 1, 2])
    renamed, _ = select(rename[matrix], np.zeros(4, dtype=bool), min_nhits=2)
    np.testing.assert_array_equal(renamed, rename[actual])


def test_witness_reports_only_shapes():
    matrix, mask = AbstractArray(shape=(5, 3), dtype='int64'), AbstractArray(shape=(8,), dtype='bool')
    selected, claimed = witness_greedy_track_commit(matrix, mask, 2, 1, .5, 3)
    assert selected.shape == (5, 3) and selected.dtype == 'int64'
    assert claimed.shape == (8,) and claimed.dtype == 'bool'

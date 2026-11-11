import numpy as np
import pytest

from sciona.tgs_mosaic import propagate_masks


def stripe(column):
    mask = np.zeros((101, 101), dtype=bool)
    mask[:, column] = True
    return mask


def test_full_vertical_source_upward_boundary_and_downward_propagation():
    train = np.stack([stripe(5)])
    query = np.zeros((4, 101, 101), dtype=bool)
    result = propagate_masks(train, [True], query, [np.array([[1], [2], [3], [0], [4]])])
    assert result['replaced_query_indices'] == [2, 3]
    np.testing.assert_array_equal(result['masks'][2:], np.stack([train[0], train[0]]))
    assert not query.any()
    early = propagate_masks(train, [True], query, [np.array([[1], [2], [0], [3]])])
    assert early['replaced_query_indices'] == [2]


def test_soft_vertical_down_only_and_first_candidate_wins():
    soft = stripe(7)
    soft[:51] = False
    train = np.stack([soft, stripe(12)])
    result = propagate_masks(train, [True, True], np.zeros((2, 101, 101)),
                             [np.array([[-1], [-1], [2], [0], [1], [3]])])
    assert result['replaced_query_indices'] == [1]
    np.testing.assert_array_equal(result['masks'][1], stripe(7))


def test_ordered_overlap_last_wins_and_ineligible_masks_ignored():
    train = np.stack([stripe(2), stripe(9), np.ones((101, 101))])
    grids = [np.array([[0], [3]]), np.array([[1], [3]]), np.array([[2], [3]])]
    result = propagate_masks(train, [True, True, True], np.zeros((1, 101, 101)), grids)
    np.testing.assert_array_equal(result['masks'][0], stripe(9))
    skipped = propagate_masks(train, [False, False, True], np.zeros((1, 101, 101)), grids)
    assert skipped['replaced_query_indices'] == []


@pytest.mark.parametrize('grid', [np.array([[2]]), np.array([[-2]]), np.array([[0.5]])])
def test_invalid_indices_rejected(grid):
    with pytest.raises(ValueError):
        propagate_masks(np.stack([stripe(1)]), [True], np.zeros((1, 101, 101)), [grid])

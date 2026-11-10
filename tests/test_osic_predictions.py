import numpy as np
import pytest
from sciona.osic_predictions import image_trajectories, quantile_predictions, blend_predictions


def test_image_trajectory_preserves_negative_uncertainty_and_query_order():
    result = image_trajectories([np.array([1., 2., 3.]), np.array([-4., -2.])],
        np.array([5., 7.]), np.array([10., 20.]), np.array([1., 2.]),
        np.array([1, 0, 0]), np.array([9., 5., 9.]))
    np.testing.assert_array_equal(result, [[14., 8.], [10., 1.], [18., -7.]])


def test_raw_crossed_quantiles_and_known_outcome_override_before_blend():
    folds = np.array([[[3., 2., 1.], [4., 5., 6.]], [[5., 4., 3.], [8., 9., 10.]]])
    original = folds.copy()
    q = quantile_predictions(folds, np.array([False, True]), np.array([0., 13.]))
    np.testing.assert_array_equal(q, [[3., -2.], [13., .1]])
    keys, result = blend_predictions(['b', 'a'], np.array([[9., 4.], [5., 2.]]), ['a', 'b'], q)
    assert keys == ['a', 'b']
    np.testing.assert_array_equal(result, .45*np.array([[5., 2.], [9., 4.]]) + .55*q)
    assert result[1, 1] != .1  # The source override applies to one branch, before blending.
    np.testing.assert_array_equal(folds, original)


def test_branch_permutations_preserve_aligned_outputs():
    x = np.array([[1., -2.], [3., 4.], [5., 6.]])
    y = np.array([[7., 8.], [9., 10.], [11., 12.]])
    keys, expected = blend_predictions(['a', 'b', 'c'], x, ['a', 'b', 'c'], y)
    actual_keys, actual = blend_predictions(['c', 'a', 'b'], x[[2, 0, 1]], ['b', 'c', 'a'], y[[1, 2, 0]])
    assert actual_keys == keys
    np.testing.assert_array_equal(actual, expected)


def test_float32_fold_division_precedes_float64_accumulation():
    smallest = np.nextafter(np.float32(0), np.float32(1))
    folds = np.full((2, 1, 3), smallest, dtype=np.float32)
    actual = quantile_predictions(folds, np.array([False]), np.array([0.]))
    np.testing.assert_array_equal(actual, [[0., 0.]])


@pytest.mark.parametrize('keys', [['a', 'a'], ['a', 'c']])
def test_reject_unaligned_or_duplicate_keys(keys):
    with pytest.raises(ValueError):
        blend_predictions(['a', 'b'], np.ones((2, 2)), keys, np.ones((2, 2)))


def test_single_slice_cannot_silently_fill_missing_source_prediction():
    with pytest.raises(ValueError):
        image_trajectories([np.array([1.])], np.array([0.]), np.array([2.]),
            np.array([1.]), np.array([0]), np.array([1.]))


def test_reject_nonfinite_fold_or_invalid_availability_mask():
    with pytest.raises(ValueError):
        quantile_predictions(np.full((2, 1, 3), np.nan), np.array([False]), np.array([0.]))
    with pytest.raises(ValueError):
        quantile_predictions(np.ones((2, 1, 3)), np.array([1]), np.array([0.]))

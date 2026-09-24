import numpy as np
import pytest

from sciona.event_count_windows import event_count_windows


def inputs():
    return [np.array([-10, -5, 0, 0, 5], dtype=np.int64),
            np.array([[-10, -10], [-5, -5], [0, 10], [10, 0], [5, 5]], dtype=np.int64),
            np.array([[True, True], [True, False], [True, True], [True, True], [True, True]]),
            np.array([0, 10, 0], dtype=np.int64), np.array([10, 20], dtype=np.int64)]


def test_channel_availability_boundaries_and_duplicate_queries():
    result = event_count_windows(*inputs())
    np.testing.assert_array_equal(result, [[[2, 3], [1, 2]], [[1, 4], [1, 3]], [[2, 3], [1, 2]]])


def test_ties_shuffle_and_future_observations_do_not_change_past():
    args = inputs()
    expected = event_count_windows(*args)
    order = np.array([3, 0, 4, 2, 1])
    np.testing.assert_array_equal(event_count_windows(args[0][order], args[1][order], args[2][order], *args[3:]), expected)
    args[0][-1] = 100
    args[1][-1] = [200, 200]
    np.testing.assert_array_equal(event_count_windows(*args)[[0, 2]], expected[[0, 2]])


def test_masked_placeholders_and_inputs_are_preserved():
    args = inputs()
    expected = event_count_windows(*args)
    args[1][~args[2]] = np.iinfo(np.int64).max
    originals = [value.copy() for value in args]
    np.testing.assert_array_equal(event_count_windows(*args), expected)
    for actual, original in zip(args, originals):
        np.testing.assert_array_equal(actual, original)


def test_empty_event_stream():
    args = inputs()
    result = event_count_windows(args[0][:0], args[1][:0], args[2][:0], *args[3:])
    np.testing.assert_array_equal(result, np.zeros((3, 2, 2), dtype=np.int64))


def test_synthetic_machine_start_and_stop_reuse():
    # Two event channels can represent independently observed machine transitions.
    result = event_count_windows(np.array([1, 4], dtype=np.int64),
        np.array([[1, 3], [4, 8]], dtype=np.int64), np.ones((2, 2), dtype=bool),
        np.array([2, 5, 8], dtype=np.int64), np.array([10], dtype=np.int64))
    np.testing.assert_array_equal(result[:, :, 0], [[1, 0], [2, 1], [2, 2]])


@pytest.mark.parametrize('slot,value', [(0, np.array([1.0])), (1, np.array([1], dtype=np.int64)),
    (2, np.ones((5, 2), dtype=np.int64)), (4, np.array([0], dtype=np.int64))])
def test_invalid_contract(slot, value):
    args = inputs()
    args[slot] = value
    with pytest.raises(ValueError):
        event_count_windows(*args)

import numpy as np
import pytest
from sciona.atoms.numerical.statistics.threshold_sampling import (
    repeat_rows_by_threshold, sequential_softmax_indices,
    witness_repeat_rows_by_threshold, witness_sequential_softmax_indices,
)
from sciona.ghost.abstract import AbstractArray
from sciona.dsb_components import oversample_by_size, sample_proposals


def same_state(a, b):
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2:] == b[2:]


def test_thresholds_are_strict_additive_and_unit_agnostic():
    rows = np.array([[-4, 10], [-2, 11], [0, 12], [2, 13], [3, 14]], dtype=np.int32)
    before = rows.copy()
    result = repeat_rows_by_threshold(rows, 0, [-2, 2], [2, 3])
    np.testing.assert_array_equal(result, rows[[2, 2, 3, 3, 4, 4, 4, 4, 4]])
    assert result.dtype == rows.dtype and not np.shares_memory(result, rows)
    np.testing.assert_array_equal(rows, before)
    assert repeat_rows_by_threshold(rows[:0], 0, [-2], [1]).shape == (0, 2)
    assert repeat_rows_by_threshold(rows, 0, [-2], [0]).shape == (0, 2)


def test_source_repetition_boundaries_match_retained_implementation():
    rows = np.column_stack((np.arange(8), [5, 6, 7, 30, 31, 40, 41, 80]))
    actual = repeat_rows_by_threshold(rows, 1, [6, 30, 40], [1, 2, 4])
    np.testing.assert_array_equal(actual, oversample_by_size(rows, 1))
    np.testing.assert_array_equal(actual[:, 0], [2, 3, 4, 4, 4, 5, 5, 5] + [6]*7 + [7]*7)


def test_each_draw_recomputes_floor_and_renormalizes():
    class Recorder(np.random.RandomState):
        def __init__(self):
            super().__init__(0)
            self.probabilities = []
        def choice(self, a, size, replace, p):
            self.probabilities.append(p.copy())
            return np.array([2 if a == 3 else 0])
    rng = Recorder()
    actual = sequential_softmax_indices(np.log([1., 2., 3.]), 2, probability_floor=.4, rng=rng)
    np.testing.assert_array_equal(actual, [2, 0])
    np.testing.assert_allclose(rng.probabilities[0], [4/13, 4/13, 5/13])
    np.testing.assert_allclose(rng.probabilities[1], [3/8, 5/8])


@pytest.mark.parametrize('seed', [0, 7, 91])
@pytest.mark.parametrize('scores', [[0., -1000., -2000.], [2., 4., -3., 1.], [0.]*5])
def test_source_sampling_and_rng_continuity(seed, scores):
    scores = np.array(scores)
    for k in [0, 1, len(scores)-1, len(scores), len(scores)+1]:
        reference, candidate = np.random.RandomState(seed), np.random.RandomState(seed)
        expected = sample_proposals(scores, k, .7, rng=reference)
        actual = sequential_softmax_indices(scores, k, .7, rng=candidate)
        np.testing.assert_array_equal(actual, expected)
        same_state(reference.get_state(), candidate.get_state())


def test_extreme_finite_scores_empty_and_global_rng_isolation():
    state = np.random.get_state()
    values = np.array([np.finfo(float).max, -np.finfo(float).max, 0.])
    actual = sequential_softmax_indices(values, 2, temperature=1e-300, random_state=3)
    assert actual[0] == 0 and len(set(actual)) == 2
    assert sequential_softmax_indices(np.array([]), 4).shape == (0,)
    same_state(state, np.random.get_state())
    rng = np.random.RandomState(6); before = rng.get_state()
    np.testing.assert_array_equal(sequential_softmax_indices(values, 4, rng=rng), [0, 1, 2])
    same_state(before, rng.get_state())


@pytest.mark.parametrize('overrides', [dict(k=True), dict(k=-1), dict(temperature=0),
    dict(temperature=True), dict(probability_floor=0), dict(probability_floor=1.1),
    dict(random_state=-1), dict(random_state=2**32), dict(rng=np.random.default_rng(0)),
    dict(random_state=0, rng=np.random.RandomState(0))])
def test_invalid_sampling_controls(overrides):
    args = dict(scores=np.array([1., 2.]), k=1); args.update(overrides)
    with pytest.raises(ValueError):sequential_softmax_indices(**args)


@pytest.mark.parametrize('scores', [[np.nan], [np.inf], [[1.]], [True], [1j]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):sequential_softmax_indices(np.asarray(scores), 1)


@pytest.mark.parametrize('thresholds,multiplicities', [([1, 1], [1, 2]), ([2, 1], [1, 2]),
    ([np.nan], [1]), ([1], [True]), ([1], [-1]), ([1], [1.5]), ([1], []),
    ([1], [2**64]), ([], [])])
def test_invalid_repetition(thresholds, multiplicities):
    with pytest.raises(ValueError):repeat_rows_by_threshold(np.array([[3.]]), 0, thresholds, multiplicities)


def test_witnesses_do_not_invent_data_dependent_cardinality():
    with pytest.raises(ValueError, match='concrete'):
        witness_repeat_rows_by_threshold(AbstractArray(shape=(4, 2), dtype='float64'), 0, [1], [2])
    output = witness_sequential_softmax_indices(AbstractArray(shape=(3,), dtype='float64'), 8)
    assert output.shape == (3,) and output.dtype == 'int64'

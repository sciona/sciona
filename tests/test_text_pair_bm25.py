"""Synthetic independent BM25 arithmetic and training-boundary checks."""
import math
import pytest
from sciona.bm25_scoring import bm25_score
from sciona.text_pair_bm25 import PairBM25


def test_numeric_hand_calculation_and_empty_query():
    # dl == average: denominator tf + k1, irrespective of b.
    expected = math.log(1 + 2.5 / 1.5) * (2 * 2.2 / 3.2)
    assert bm25_score([2, 0], [1, 2], 3, 3, 3., 1.2, .75) == pytest.approx(expected)
    assert bm25_score([], [], 0, 3, 3., 1.2, 1.) == 0
    assert bm25_score([0], [0], 0, 3, 3., 1.2, 1.) == 0


def test_length_normalization_and_frequency_saturation():
    def score(tf, length, b=.75):
        return bm25_score([tf], [2], length, 10, 5., 1.2, b)
    assert score(1, 10) < score(1, 5) < score(1, 2)
    assert score(1, 5, 0) == score(1, 10, 0)
    assert score(3, 5) - score(2, 5) < score(2, 5) - score(1, 5)


@pytest.mark.parametrize('args', [
    ([1], [], 1, 2, 1., 1.2, .75), ([True], [1], 1, 2, 1., 1.2, .75),
    ([2], [1], 1, 2, 1., 1.2, .75), ([1], [3], 1, 2, 1., 1.2, .75),
    ([1], [-1], 1, 2, 1., 1.2, .75), ([1], [1], 1, 0, 1., 1.2, .75),
    ([1], [1], 1, 2, 0., 1.2, .75), ([1], [1], 1, 2, 1., 0., .75),
    ([1], [1], 1, 2, 1., 1.2, 1.01), ([1], [1], 1, 2, math.nan, 1.2, .75),
    ([1], [1], 1, 2, 1., math.inf, .75), ([1], [1], 1, 2, 1., True, .75),
    ([1], [1], 1, 2, 5e-324, 1.2, .75),
    ([1], [1], 1, 2, 1., 10**400, .75),
])
def test_invalid_counts_controls_and_overflow(args):
    with pytest.raises(ValueError):
        bm25_score(*args)


def fitted():
    return PairBM25([('red red blue', 'green'), ('green', 'blue green')], k1=1.2, b=.75)


def test_unique_training_corpus_and_independent_directional_reference():
    model = fitted()
    assert model.document_count == 3
    assert model.average_length == 2
    assert dict(model.document_frequencies) == {'red': 1, 'blue': 2, 'green': 2}
    # red->red red blue: tf=2, dl=3; reverse: tf=1, dl=1.
    idf = math.log(1 + 2.5 / 1.5)
    expected = idf * (4.4 / (2 + 1.2 * (.25 + .75 * 3 / 2))
                      + 2.2 / (1 + 1.2 * (.25 + .75 / 2))) / 2
    assert model.transform([('red', 'red red blue')]) == pytest.approx([expected])


def test_symmetry_row_order_frozen_state_and_unseen_terms():
    model = fitted()
    pairs = [('red', 'red red blue'), ('novel', 'novel'), ('red', 'green')]
    scores = model.transform(pairs)
    assert model.transform([(b, a) for a, b in pairs]) == scores
    assert model.transform(list(reversed(pairs))) == list(reversed(scores))
    assert scores[1] > 0 and scores[2] == 0
    model.transform([('other other', 'novel other')])
    assert model.transform(pairs) == scores
    assert 'novel' not in model.document_frequencies
    with pytest.raises(TypeError):
        model.document_frequencies['red'] = 99


def test_training_input_mutation_and_duplicate_endpoints_do_not_change_fit():
    pairs = [['red red blue', 'green'], ['green', 'blue green']]
    model = PairBM25(pairs * 2, k1=1.2, b=.75)
    pairs[0][0] = 'changed'
    assert model == fitted()


def test_normalization_policy_then_frozen_training_statistics():
    from sciona.text_pair_normalization_policy import NormalizationPolicy
    policy = NormalizationPolicy({'colour': 10, 'cat': 8, 'pet': 7, 'animal': 6},
                                 {'colour': ['color']}, {'cat': ['pet']})
    training, calibration, query = policy.normalize_populations(
        [['colour', 'cat']], [['pet', 'animal']], [['animal', 'animal']])
    model = PairBM25(training, k1=1.2, b=.75)
    assert dict(model.document_frequencies) == {'color': 1, 'cat': 1, 'pet': 1}
    before = dict(model.document_frequencies)
    assert len(model.transform(calibration)) == len(calibration)
    assert len(model.transform(query)) == len(query)
    assert dict(model.document_frequencies) == before


@pytest.mark.parametrize('pairs', [[], [('UPPER', 'word')], [('two  spaces', 'word')],
                                 [('word!', 'word')], [('', 'word')], [('word',)],
                                 [('x' * 513, 'word')]])
def test_reject_unnormalized_inputs_without_silent_second_normalization(pairs):
    with pytest.raises(ValueError):
        fitted().transform(pairs)

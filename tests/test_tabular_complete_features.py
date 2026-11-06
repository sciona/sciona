"""Synthetic integration checks for cleaning and excluded target features."""
import copy

import numpy as np
import pytest

from sciona.tabular_complete_features import fit_binary_tabular_features


CONTROLS = dict(clip_low=0., clip_high=1., max_mode_fraction=1., log_columns=[], interactions=[], smoothing=0., group_aggregates=[], text_topics=[])


def test_aggregate_features_use_training_cleaning_and_frozen_keys():
    spec = dict(key_column=0, value_columns=[0], statistics=['mean', 'count'], unknown_value=-1.)
    state, training = fit_binary_tabular_features([[2.], [None], [10.], [6.]],
        [['a'], ['a'], ['b'], ['b']], [0, 1, 0, 1], ['g0', 'g1', 'g2', 'g3'],
        **dict(CONTROLS, group_aggregates=[spec]))
    # Training median is 6; a has mean 4, b has mean 8.
    np.testing.assert_array_equal(training[:, -2:], [[4, 2], [4, 2], [8, 2], [8, 2]])
    query = state.transform([[1000.], [-1000.], [None]], [['a'], ['b'], ['new']])
    np.testing.assert_array_equal(query[:, -2:], [[4, 2], [8, 2], [-1, -1]])


@pytest.mark.parametrize('spec', [
    {}, dict(key_column=9, value_columns=[0], statistics=['mean'], unknown_value=0),
    dict(key_column=0, value_columns=[0, 0], statistics=['mean'], unknown_value=0),
])
def test_rejects_invalid_aggregate_policy(spec):
    with pytest.raises(ValueError):
        fit_binary_tabular_features([[1.], [2.]], [['a'], ['b']], [0, 1], ['g0', 'g1'],
            **dict(CONTROLS, group_aggregates=[spec]))


def fit(numeric, categorical, labels=None, groups=None, **controls):
    return fit_binary_tabular_features(numeric, categorical, labels or [0, 1, 0, 1],
        groups or ['a', 'a', 'b', 'c'], **dict(CONTROLS, **controls))


def test_training_target_features_are_distinct_from_full_fit_query_features():
    n = [[0.], [1.], [2.], [3.]]; c = [['x'], ['x'], ['y'], ['z']]
    state, training = fit(n, c)
    np.testing.assert_array_equal(training[:2, -1], [.5, .5])
    assert training[2, -1] == pytest.approx(2/3)
    assert training[3, -1] == pytest.approx(1/3)
    np.testing.assert_array_equal(state.transform(n, c)[:, -1], [.5, .5, 0., 1.])


def test_fold_local_screening_logs_and_interactions_keep_original_column_policy():
    n = [[0., 9.], [0., 9.], [0., 9.], [8., 9.]]
    c = [['x', 'x', 'fixed'], ['y', 'y', 'fixed'], ['x', 'x', 'fixed'], ['z', 'z', 'fixed']]
    before = copy.deepcopy((n, c))
    state, training = fit(n, c, max_mode_fraction=.7, log_columns=[0], interactions=[[0, 1], [0, 2]])
    assert state.categorical_columns == (0,)
    assert state.categories.interactions == ()
    # Both numeric variants have modal fraction .75 and are removed; three
    # one-hot columns plus the count/frequency/target triplet remain.
    assert training.shape == (4, 6)
    assert (n, c) == before
    alone = state.transform([[4., 0.]], [['x', 'ignored', 'ignored']])
    batch = state.transform([[4., 0.], [999., -3.]], [['x', 'ignored', 'ignored'], ['new', 'x', 'fixed']])
    np.testing.assert_array_equal(alone[0], batch[0])


def test_numeric_only_signed_log_and_missing_imputation():
    state, training = fit([[-3.], [0.], [3.], [None]], [[], [], [], []], log_columns=[0])
    assert state.categories is None
    np.testing.assert_allclose(training[:, 1], [-np.log(4), 0., np.log(4), 0.])
    np.testing.assert_array_equal(state.transform([[None]], [[]]), [[0., 0.]])


def test_valid_interaction_and_unknown_tokens_preserve_order():
    n = [[], [], [], []]; c = [['x', 'p'], ['x', 'q'], ['y', 'p'], ['y', 'q']]
    state, training = fit(n, c, interactions=[[0, 1]])
    assert training.shape == (4, 13)  # four one-hot columns and three triplets
    result = state.transform([[], []], [['unknown', 'p'], ['unknown', 'p']])
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(result[:, -3:], [[0., 0., .5], [0., 0., .5]])


@pytest.mark.parametrize('controls', [dict(max_mode_fraction=0), dict(max_mode_fraction=True),
    dict(max_mode_fraction=float('nan')), dict(log_columns=[0, 0]), dict(log_columns=[True]),
    dict(log_columns=[1]), dict(smoothing=-1), dict(interactions=[[0, 0]])])
def test_bad_policies_reject(controls):
    with pytest.raises(ValueError):
        fit([[0.], [1.], [2.], [3.]], [['x'], ['x'], ['y'], ['z']], **controls)

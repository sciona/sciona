"""Synthetic cross-disciplinary targets, exact counts and label-isolation tests."""
import copy

import numpy as np
import pytest

from sciona.tabular_category_features import fit_category_features


def fit(rows=None, targets=None, groups=None, **kwargs):
    return fit_category_features(rows if rows is not None else [['a'], ['a'], ['b'], ['b']],
        targets if targets is not None else [10., 30., -10., 50.],
        groups if groups is not None else ['g1', 'g1', 'g2', 'g3'], interactions=kwargs.get('interactions', []), smoothing=kwargs.get('smoothing', 2.))


def test_counts_frequencies_and_group_excluded_unknown_prior_by_hand():
    state, training = fit()
    np.testing.assert_array_equal(training[:, :2], [[2., .5]]*4)
    # Entire g1 is absent from its fitting population, including the prior.
    np.testing.assert_array_equal(training[:2, 2], [20., 20.])
    assert training[2, 2] == pytest.approx((50.+2.*30.)/3.)
    assert training[3, 2] == pytest.approx((-10.+2.*10.)/3.)
    np.testing.assert_array_equal(state.transform([['unknown'], ['a'], ['a']]),
        [[0., 0., 20.], [2., .5, 20.], [2., .5, 20.]])


def test_changing_group_targets_cannot_change_its_training_features():
    _, before = fit()
    _, after = fit(targets=[1000., -900., -10., 50.])
    np.testing.assert_array_equal(before[:2], after[:2])
    assert not np.array_equal(before[2:, 2], after[2:, 2])


def test_interactions_are_collision_free_and_inputs_and_maps_remain_unchanged():
    rows = [['a|b', 'c'], ['a', 'b|c'], ['a|b', 'c'], ['', '"\\']]
    original = copy.deepcopy(rows)
    state, training = fit(rows, groups=['g1', 'g2', 'g3', 'g4'], interactions=[[0, 1]])
    assert training.shape == (4, 9)
    np.testing.assert_array_equal(training[:, 6], [2., 1., 2., 1.])
    snapshot = tuple(dict(m) for m in state.target_means)
    state.transform([['novel', 'query'], ['a', 'b|c']])
    assert snapshot == tuple(dict(m) for m in state.target_means)
    assert rows == original
    with pytest.raises(TypeError):
        state.target_means[0]['a'] = 99.


@pytest.mark.parametrize('kwargs', [dict(targets=[True, 0, 1, 0]), dict(targets=[float('nan')]*4),
    dict(groups=['same']*4), dict(groups=['a', 'b']), dict(interactions=[[0, 0]]),
    dict(interactions=[[True, 0]]), dict(interactions=[[0, 1]]), dict(smoothing=-1.),
    dict(smoothing=True), dict(smoothing=float('inf')), dict(rows=[['a'], [None], ['a'], ['b']])])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        fit(**kwargs)


def test_zero_smoothing_matches_excluded_group_category_mean():
    _, values = fit([['x'], ['x'], ['x'], ['x']], smoothing=0.)
    np.testing.assert_array_equal(values[:, 2], [20., 20., 30., 10.])

"""Synthetic selection behavior and outer-fold isolation evidence."""
import numpy as np
import pytest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sciona.tabular_feature_selection import select_training_features
from sciona.tabular_complete_training import fit_complete_stacker
from tests.test_tabular_complete_training import POLICY
from tests.test_tabular_ensemble_training import sample


class FrozenFeatures:
    def transform(self, numeric, categorical):
        return np.asarray(numeric, dtype=np.float64)


def test_one_remaining_feature_requires_no_elimination():
    x = np.array([[0.], [1.], [2.], [3.]])
    state, training = select_training_features(FrozenFeatures(), x, [0, 1, 0, 1],
        policy=dict(method='rfe', keep=1, step=1), seed=42)
    assert state.columns == (0,) and state.ranking == (1,)
    np.testing.assert_array_equal(training, x)


def test_selection_matches_sklearn_and_freezes_query_columns():
    x = np.array([[i, i % 3, (-1)**i, i % 5] for i in range(20)], dtype=float)
    y = np.array([i >= 10 for i in range(20)], dtype=int)
    state, training = select_training_features(FrozenFeatures(), x, y,
        policy=dict(method='rfe', keep=2, step=1), seed=42)
    reference = RFE(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        n_features_to_select=2, step=1, importance_getter='named_steps.logisticregression.coef_').fit(x, y)
    np.testing.assert_array_equal(training, reference.transform(x))
    np.testing.assert_array_equal(state.ranking, reference.ranking_)
    query = [[-100, 9, 20, 500], [0, 0, 0, 0]]
    np.testing.assert_array_equal(state.transform(query, [[], []]), reference.transform(query))
    assert len(state.columns) == 2


def test_outer_validation_mutations_do_not_change_selection_or_fit():
    n, c, y, folds, groups = sample()
    policy = dict(POLICY, selection=dict(method='rfe', keep=2, step=2))
    first = fit_complete_stacker(n, c, y, folds, groups, feature_controls=policy, trees=8)
    changed_n = [row[:] for row in n]; changed_c = [row[:] for row in c]; changed_y = y[:]
    for i, fold in enumerate(folds):
        if fold == 0:
            changed_n[i] = [1000., -2000.]; changed_c[i] = ['heldout']; changed_y[i] = 1-y[i]
    second = fit_complete_stacker(changed_n, changed_c, changed_y, folds, groups,
        feature_controls=policy, trees=8)
    a, b = first.fold_models[0], second.fold_models[0]
    assert a.features.columns == b.features.columns
    assert a.features.ranking == b.features.ranking
    np.testing.assert_array_equal(a.linear[-1].coef_, b.linear[-1].coef_)
    np.testing.assert_array_equal(a.predict(n[:4], c[:4]), b.predict(n[:4], c[:4]))
    assert all(model.linear.n_features_in_ == 2 for model in [first.base, *first.fold_models.values()])


def test_selection_consumes_supplied_training_features_without_retransform():
    class NoTrainingTransform:
        def transform(self, *args):
            raise AssertionError('Training must use the supplied excluded matrix')
    x = np.array([[i, i % 2, i % 3] for i in range(12)], dtype=float)
    state, training = select_training_features(NoTrainingTransform(), x, np.arange(12) % 2,
        policy=dict(method='rfe', keep=1, step=1), seed=42)
    np.testing.assert_array_equal(training, x[:, state.columns])
    assert state.columns == (1,)


@pytest.mark.parametrize('policy', [
    {}, dict(method='unknown', keep=1, step=1), dict(method='rfe', keep=0, step=1),
    dict(method='rfe', keep=4, step=1), dict(method='rfe', keep=True, step=1),
    dict(method='rfe', keep=1, step=0), dict(method='rfe', keep=1, step=0.5),
])
def test_invalid_selection_policy(policy):
    with pytest.raises(ValueError):
        select_training_features(FrozenFeatures(), np.ones((4, 3)), [0, 1, 0, 1], policy=policy, seed=42)

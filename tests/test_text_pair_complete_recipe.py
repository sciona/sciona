"""Synthetic complete recipe integration and held-out fitting boundaries."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from sciona.text_pair_complete_recipe import execute, fit_normalized
from sciona.text_pair_normalization_policy import NormalizationPolicy
from tests.test_text_pair_training import data


CONTROLS = dict(components=2, seed=42, iterations=5, oversamples=10,
                k1=1.2, b=.75, trees=16, max_depth=6, min_leaf=1)


def policy():
    # Explicit synthetic vocabulary includes held-out terms by design; no fit.
    words = 'red bird blue lake green tree small cat wide river kitten trees purple cloud colour pet'
    return NormalizationPolicy({word: 1 for word in words.split()},
                               {'colour': ['red']}, {'cat': ['pet']})


def test_full_recipe_reproducible_with_independent_f1_selection():
    training, y, cal, cy, query = data()
    result = execute(training, y, cal, cy, query, policy=policy(), **CONTROLS)
    assert result == execute(training, y, cal, cy, query, policy=policy(), **CONTROLS)
    train_n, cal_n, query_n = policy().normalize_populations(training, cal, query)
    fitted = fit_normalized(train_n, y, **CONTROLS)
    scores = fitted.scores(cal_n)
    expected_f1, expected_threshold = max((f1_score(cy, scores >= t), t) for t in np.unique(scores))
    assert result['threshold'] == expected_threshold
    assert result['calibration_f1'] == expected_f1
    assert result['scores'] == fitted.scores(query_n).tolist()
    assert result['matches'] == [int(s >= expected_threshold) for s in result['scores']]
    assert result['feature_count'] == 13 and result['query_rows'] == 2
    assert len(fitted.forest.estimators_) == CONTROLS['trees']


def test_training_only_state_and_bm25_feature_survive_held_out_transforms():
    training, y, cal, _, query = data()
    train_n, cal_n, query_n = policy().normalize_populations(training, cal, query)
    fitted = fit_normalized(train_n, y, **CONTROLS)
    features = fitted.features
    vocab = dict(features.vectorizer.vocabulary_)
    components = features.svd.components_.copy()
    frequencies = dict(features.bm25.document_frequencies)
    coefficients = fitted.linear[-1].coef_.copy()
    expected = fitted.scores(query_n)
    fitted.scores(cal_n)
    repeated = query_n[::-1] + [query_n[0]]
    np.testing.assert_array_equal(fitted.scores(repeated), np.r_[expected[::-1], expected[0]])
    np.testing.assert_array_equal(components, features.svd.components_)
    np.testing.assert_array_equal(coefficients, fitted.linear[-1].coef_)
    assert frequencies == dict(features.bm25.document_frequencies)
    assert vocab == features.vectorizer.vocabulary_
    assert 'purple' not in vocab and 'purple' not in frequencies
    np.testing.assert_array_equal(features.transform(query_n)[:, -1], features.bm25.transform(query_n))
    # Existing graph context excludes each current training edge.
    assert np.all(features.transform_training()[:, 11] == 0)


def test_policy_collision_rejects_before_training(monkeypatch):
    import sciona.text_pair_complete_recipe as recipe
    def forbidden(*args, **kwargs):
        pytest.fail('Overlapping populations reached training')
    monkeypatch.setattr(recipe, 'fit_normalized', forbidden)
    training, y, _, _, query = data()
    with pytest.raises(ValueError, match='Overlapping'):
        execute(training, y, [['bird colour', 'red bird'], ['small cat', 'small kitten']],
                [0, 1], query, policy=policy(), **CONTROLS)


def test_calibration_labels_do_not_change_query_scores():
    training, y, cal, cy, query = data()
    first = execute(training, y, cal, cy, query, policy=policy(), **CONTROLS)
    second = execute(training, y, cal, [1-v for v in cy], query, policy=policy(), **CONTROLS)
    assert first['scores'] == second['scores']


@pytest.mark.parametrize('control,value', [('trees', 0), ('min_leaf', True), ('components', 0), ('b', 2)])
def test_invalid_controls(control, value):
    training, y, cal, cy, query = data()
    controls = dict(CONTROLS, **{control: value})
    with pytest.raises(ValueError):
        execute(training, y, cal, cy, query, policy=policy(), **controls)

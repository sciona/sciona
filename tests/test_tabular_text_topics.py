"""Synthetic text-topic behavior and fold-isolation checks."""
import numpy as np
import pytest

from sciona.tabular_text_topics import fit_text_topics
from sciona.tabular_complete_training import fit_complete_stacker
from tests.test_tabular_complete_training import POLICY
from tests.test_tabular_ensemble_training import sample


CONTROLS = dict(topics=2, max_features=20, iterations=4, lowercase=False, seed=42)
DOCUMENTS = ['red green red', 'blue blue', 'green red', 'blue yellow']


def test_topics_match_model_normalization_and_freeze_vocabulary():
    state, training = fit_text_topics(DOCUMENTS, **CONTROLS)
    np.testing.assert_allclose(training, state.model.transform(state.vectorizer.transform(DOCUMENTS)), rtol=0, atol=0)
    np.testing.assert_allclose(training.sum(axis=1), 1.)
    vocabulary = dict(state.vectorizer.vocabulary_); components = state.model.components_.copy()
    query = state.transform(['unseen', '', 'red green', 'red green'])
    np.testing.assert_allclose(query[:2], .5)
    np.testing.assert_array_equal(query[2], query[3])
    assert state.vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(state.model.components_, components)
    assert state.model.n_jobs == 1


def test_case_and_whitespace_policy_and_repeatability():
    a, x = fit_text_topics(['A a a!', 'B b'], **CONTROLS)
    b, y = fit_text_topics(['A a a!', 'B b'], **CONTROLS)
    assert set(a.vectorizer.vocabulary_) == {'A', 'a', 'a!', 'B', 'b'}
    np.testing.assert_array_equal(x, y)
    c, _ = fit_text_topics(['A a a!', 'B b'], **dict(CONTROLS, lowercase=True))
    assert set(c.vectorizer.vocabulary_) == {'a', 'a!', 'b'}


def test_outer_validation_text_cannot_fit_vocabulary_or_topics():
    n, c, y, folds, groups = sample()
    c = [[DOCUMENTS[i % 4]] for i in range(len(n))]
    policy = dict(POLICY, text_topics=[dict(column=0, **CONTROLS)])
    first = fit_complete_stacker(n, c, y, folds, groups, feature_controls=policy, trees=8)
    changed = [row[:] for row in c]
    for i, fold in enumerate(folds):
        if fold == 0:
            changed[i] = ['heldoutonly newword']
    second = fit_complete_stacker(n, changed, y, folds, groups, feature_controls=policy, trees=8)
    a = first.fold_models[0].features.text_topics[0][1]
    b = second.fold_models[0].features.text_topics[0][1]
    assert a.vectorizer.vocabulary_ == b.vectorizer.vocabulary_
    assert 'heldoutonly' not in b.vectorizer.vocabulary_
    np.testing.assert_array_equal(a.model.components_, b.model.components_)
    np.testing.assert_array_equal(first.fold_models[0].predict(n[:4], c[:4]),
        second.fold_models[0].predict(n[:4], c[:4]))


def test_null_text_uses_empty_document_prior_in_combined_transform():
    from sciona.tabular_complete_features import fit_binary_tabular_features
    from tests.test_tabular_complete_features import CONTROLS as FEATURE_CONTROLS
    state, _ = fit_binary_tabular_features([[1.], [2.], [3.], [4.]],
        [[text] for text in DOCUMENTS], [0, 1, 0, 1], ['a', 'b', 'c', 'd'],
        **dict(FEATURE_CONTROLS, text_topics=[dict(column=0, **CONTROLS)]))
    np.testing.assert_allclose(state.transform([[2.]], [[None]])[:, -2:], [[.5, .5]])


@pytest.mark.parametrize('change', [dict(topics=0), dict(topics=True), dict(max_features=-1),
    dict(iterations=0), dict(lowercase=1), dict(seed=-1)])
def test_invalid_controls(change):
    with pytest.raises(ValueError):
        fit_text_topics(DOCUMENTS, **dict(CONTROLS, **change))


@pytest.mark.parametrize('documents', [[], [None], ['', '  ']])
def test_invalid_or_empty_training_vocabulary(documents):
    with pytest.raises(ValueError):
        fit_text_topics(documents, **CONTROLS)

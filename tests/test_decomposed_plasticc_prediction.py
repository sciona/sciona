"""Synthetic numeric inference contracts independent of astronomy labels."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sciona.decomposed_plasticc_prediction import normalize_scores, predict_scores, average_folds


def test_all_models_source_order_and_no_mutation():
    matrices = [np.array([[1., 2., 3.], [3., 0., 1.]]) + i * np.array([1., -1., 0.]) for i in range(4)]
    originals = [value.copy() for value in matrices]
    features = pd.DataFrame({'feature': [1., 2.]}, index=['synthetic-a', 'synthetic-b'])
    classes = ['x', 'y', 'z']; calls = []
    models = []
    for index, scores in enumerate(matrices):
        def infer(values, *, raw_score, num_iteration, index=index, scores=scores):
            assert values is features and raw_score and num_iteration == index + 2
            calls.append(index)
            return scores
        models.append(SimpleNamespace(classes_=np.array(classes), best_iteration_=index + 2, predict_proba=infer))
    expected = 0
    for scores in matrices:
        exp = np.exp(scores)
        expected += exp / exp.sum(axis=1)[:, None]
    expected /= len(models)
    actual = average_folds(models, features, classes, predict_scores, normalize_scores)
    np.testing.assert_array_equal(actual.to_numpy(), expected)
    assert calls == [0, 1, 2, 3]
    assert list(actual.index) == list(features.index) and list(actual.columns) == classes
    for original, matrix in zip(originals, matrices):
        np.testing.assert_array_equal(original, matrix)


@pytest.mark.parametrize('scores', [[], [1., 2.], [[np.nan, 1]], [[np.inf, 1]], [[1000., 999.]], [[-1000., -999.]]])
def test_invalid_or_nonfinite_normalization_rejected(scores):
    with pytest.raises(ValueError):
        normalize_scores(scores)


def test_class_order_rejected_before_inference():
    def forbidden(*args, **kwargs):
        pytest.fail('inference must not run for mismatched classes')
    model = SimpleNamespace(classes_=['b', 'a'], predict_proba=forbidden)
    with pytest.raises(ValueError):
        predict_scores(model, pd.DataFrame({'f': [1]}), ['a', 'b'])

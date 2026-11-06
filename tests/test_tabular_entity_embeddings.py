"""Synthetic categorical embedding and validation-isolation evidence."""
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from sciona.tabular_entity_embeddings import fit_entity_embeddings
from sciona.tabular_complete_training import fit_complete_stacker
from tests.test_tabular_complete_training import POLICY
from tests.test_tabular_ensemble_training import sample


CONTROLS = dict(dimension=2, iterations=1000, regularization=.1, tolerance=1e-5, seed=42)
ROWS = [['a', 'x'], ['b', 'y'], ['a', 'y'], ['b', 'x']]*3
LABELS = [0, 1, 0, 1]*3


def test_embedding_vectors_match_trained_network_input_weights():
    state, training = fit_entity_embeddings(ROWS, LABELS, **CONTROLS)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
    x = encoder.fit_transform(ROWS)
    model = MLPClassifier(hidden_layer_sizes=(2,), activation='tanh', solver='lbfgs',
        alpha=.1, max_iter=1000, tol=1e-5, random_state=42).fit(x, LABELS)
    expected = np.column_stack([model.coefs_[0][[0, 1, 0, 1]*3], model.coefs_[0][[2, 3, 3, 2]*3]])
    np.testing.assert_array_equal(training, expected)
    # Summed per-field vectors reproduce the network's pre-bias hidden input.
    np.testing.assert_allclose(training[:, :2]+training[:, 2:], x @ model.coefs_[0])
    unknown = state.transform([['new', 'x'], ['new', 'new']])
    np.testing.assert_array_equal(unknown[:, :2], 0.)
    np.testing.assert_array_equal(unknown[1], 0.)
    with pytest.raises(TypeError):
        state.maps[0]['a'] = (0., 0.)


def test_embedding_is_supervised_repeatable_and_query_frozen():
    state, original = fit_entity_embeddings(ROWS, LABELS, **CONTROLS)
    _, repeat = fit_entity_embeddings(ROWS, LABELS, **CONTROLS)
    _, changed = fit_entity_embeddings(ROWS, [1-v for v in LABELS], **CONTROLS)
    np.testing.assert_array_equal(original, repeat)
    assert not np.allclose(original, changed)
    state.transform([['query', 'new']])
    np.testing.assert_array_equal(state.transform(ROWS), original)


def test_validation_labels_and_categories_do_not_fit_embeddings():
    n, c, y, folds, groups = sample()
    policy = dict(POLICY, entity_embeddings=[dict(columns=[0], **CONTROLS)])
    first = fit_complete_stacker(n, c, y, folds, groups, feature_controls=policy, trees=8)
    changed_c = [row[:] for row in c]; changed_y = y[:]
    for i, fold in enumerate(folds):
        if fold == 0:
            changed_c[i] = ['heldout-only']; changed_y[i] = 1-y[i]
    second = fit_complete_stacker(n, changed_c, changed_y, folds, groups, feature_controls=policy, trees=8)
    a, b = first.fold_models[0], second.fold_models[0]
    assert a.features.embeddings[0][1].maps == b.features.embeddings[0][1].maps
    np.testing.assert_array_equal(a.predict(n[:4], c[:4]), b.predict(n[:4], c[:4]))


@pytest.mark.parametrize('change', [dict(dimension=0), dict(dimension=True), dict(iterations=0),
    dict(regularization=-1), dict(tolerance=0), dict(seed=-1)])
def test_invalid_controls(change):
    with pytest.raises(ValueError):
        fit_entity_embeddings(ROWS, LABELS, **dict(CONTROLS, **change))


def test_unknown_shape_and_single_class_rejected():
    state, _ = fit_entity_embeddings(ROWS, LABELS, **CONTROLS)
    with pytest.raises(ValueError):
        state.transform([['one column']])
    with pytest.raises(ValueError):
        fit_entity_embeddings(ROWS, [0]*len(ROWS), **CONTROLS)


def test_all_feature_families_through_calibration_and_query():
    from sciona.tabular_ensemble_calibration import calibrate_stack, predict_calibrated
    n, c, y, folds, groups = sample()
    c = [[row[0], 'red green' if i % 2 else 'blue yellow'] for i, row in enumerate(c)]
    policy = dict(POLICY, interactions=[[0, 1]],
        group_aggregates=[dict(key_column=0, value_columns=[0], statistics=['mean'], unknown_value=0.)],
        text_topics=[dict(column=1, topics=2, max_features=10, iterations=4, lowercase=True, seed=42)],
        entity_embeddings=[dict(columns=[0, 1], **CONTROLS)],
        selection=dict(method='rfe', keep=4, step=3))
    model = fit_complete_stacker(n, c, y, folds, groups, feature_controls=policy, trees=8)
    assert model.oof_predictions.shape == (12, 2)
    features = model.base.features.source
    assert features.categories is not None
    assert len(features.aggregates) == len(features.text_topics) == len(features.embeddings) == 1
    calibrated = calibrate_stack(model, [[0., 0.], [1., 0.], [4., 1.], [7., 3.]],
        [['category0', 'red'], ['category1', 'blue'], ['new', 'green'], ['category2', 'yellow']],
        [0, 1, 0, 1], ['cal0', 'cal1', 'cal2', 'cal3'])
    result = predict_calibrated(calibrated, [[2., 0.], [7., None]],
        [['category0', 'unseen'], ['new', None]], ['query0', 'query1'])
    assert all(np.isfinite(p) and 0 <= p <= 1 for p in result['probabilities'])
    assert result['classes'] == [int(p >= .5) for p in result['probabilities']]

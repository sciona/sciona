"""Complete synthetic training parity, including OOF state and all folds."""
import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sciona.decomposed_plasticc_sampling import balanced_weights, reference_stratified_folds
from sciona.decomposed_plasticc_training import train, fit_folds
from sciona.plasticc_training import fit_lightgbm_classifier
from sciona.plasticc_runtime import prepare_runtime, train_runtime, predict_runtime
from sciona.plasticc_weighting import weighted_multi_logloss
from scripts.validate_plasticc_runtime import synthetic_payload
from sciona.decomposed_plasticc_prediction import predict, predict_scores, normalize_scores


@pytest.mark.parametrize('width', [2, 7])
def test_generic_width_nonastronomy_classes(width):
    from sciona.plasticc_training import LightGBMClassifier
    from sciona.atoms.ml import decomposed_plasticc_lifecycle as providers
    rng = np.random.default_rng(73)
    features = pd.DataFrame(rng.normal(size=(36, width)), columns=[f'feature_{i}' for i in range(width)])
    labels = pd.Series(np.repeat(['category-a', 'category-b', 'category-c'], 12))
    references = pd.Series(None, index=labels.index, dtype=object)
    folds = reference_stratified_folds(labels, references, 3, 17)
    weights = balanced_weights(labels, None)
    parameters = dict(n_jobs=1, n_estimators=60, min_child_weight=.01, min_child_samples=2, min_split_gain=0, random_seed=4)
    dataset = SimpleNamespace(metadata=pd.DataFrame({'class': labels}), select_features=lambda _: features,
                              label_folds=lambda *_: folds)
    expected = LightGBMClassifier('synthetic', None)
    expected.train(dataset, num_folds=3, random_state=17, **parameters)
    actual = providers.fit_folds(features, labels, weights, folds, parameters, providers.fit_fold)
    pd.testing.assert_frame_equal(actual.predictions, expected.train_predictions, check_exact=True)
    pd.testing.assert_frame_equal(actual.importances, expected.importances, check_exact=True)
    assert len(actual.models) == 3
    for left, right in zip(actual.models, expected.classifiers, strict=True):
        assert left.booster_.model_to_string() == right.booster_.model_to_string()


@pytest.mark.parametrize('weighting,class_weighting', [('flat', 'flat'), ('redshift', 'kaggle')])
def test_complete_training_parity(weighting, class_weighting):
    payload = synthetic_payload()
    payload['config'].update(weighting=weighting, class_weighting=class_weighting)
    original = copy.deepcopy(payload)
    expected = train_runtime(prepare_runtime(payload))
    calls = {'fold': 0, 'balance': 0, 'fit': 0}
    def tracked(name, fn):
        def invoke(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return invoke
    actual = train(prepare_runtime(payload), tracked('fold', reference_stratified_folds),
                   tracked('balance', balanced_weights), tracked('fit', fit_lightgbm_classifier))
    assert calls == {'fold': 1, 'balance': int(weighting == 'flat'), 'fit': payload['config']['num_folds']}
    a = actual.classifier; e = expected.classifier
    pd.testing.assert_frame_equal(a.train_predictions, e.train_predictions, check_exact=True)
    pd.testing.assert_frame_equal(a.importances, e.importances, check_exact=True)
    pd.testing.assert_series_equal(a.train_classes, e.train_classes)
    for left, right in zip(a.classifiers, e.classifiers, strict=True):
        assert left.booster_.model_to_string() == right.booster_.model_to_string()
    assert set(a.loss_diagnostics) == {'all', 'original'}
    assert a.loss_diagnostics['all'][1] == weighted_multi_logloss(e.train_classes, e.train_predictions, class_weights=e.class_weights)
    assert predict(actual, predict_scores, normalize_scores) == predict_runtime(expected)
    assert payload == original


@pytest.mark.parametrize('fault', ['workers', 'alignment', 'missing_class', 'fold_gap'])
def test_invalid_fold_execution_rejected_before_training(fault):
    features = pd.DataFrame(np.arange(12).reshape(6, 2), columns=['a', 'b'])
    labels = pd.Series([0, 1, 0, 1, 0, 1])
    weights = pd.Series(np.ones(6)); folds = pd.Series([0, 0, 1, 1, 2, 2])
    params = {'n_jobs': 1}
    if fault == 'workers': params['n_jobs'] = 4
    if fault == 'alignment': weights = weights.iloc[::-1]
    if fault == 'missing_class': labels = pd.Series([2, 2, 0, 1, 0, 1])
    if fault == 'fold_gap': folds = folds.replace(1, 3)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid contract reached training')
    with pytest.raises(ValueError):
        fit_folds(features, labels, weights, folds, params, forbidden)

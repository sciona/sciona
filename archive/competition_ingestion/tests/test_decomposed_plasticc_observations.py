"""Cache and full synthetic lifecycle checks with explicit numeric bindings."""
import copy
from unittest.mock import patch
import numpy as np
import pytest
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix
from sciona import decomposed_plasticc_gp as gp
from sciona import decomposed_plasticc_gp_fit as fitting
from sciona.decomposed_plasticc_observations import prepare, NumericObject
from sciona.plasticc_runtime import prepare_runtime, train_runtime, predict_runtime
from scripts.validate_plasticc_runtime import synthetic_payload
from sciona.decomposed_plasticc_training import train as train_decomposed
from sciona.decomposed_plasticc_sampling import balanced_weights, reference_stratified_folds
from sciona.plasticc_training import fit_lightgbm_classifier
from sciona.decomposed_plasticc_prediction import predict, predict_scores, normalize_scores

OPERATIONS = dict(amplitude=fitting.initial_amplitude, fit=fitting.fit, kernel=matern_kernel_matrix,
    scale=gp.scale_covariance, noise=gp.observation_variance, condition=gp.condition, predict=gp.predict)


def test_default_cache_and_explicit_refit():
    calls = []
    def fit(*args): calls.append(1); return fitting.fit(*args)
    prepared = prepare(synthetic_payload(), dict(OPERATIONS, fit=fit))
    obj = prepared.training.objects[0]
    first = obj.get_default_gaussian_process()
    assert obj.get_default_gaussian_process() is first and len(calls) == 1
    explicit, _, _ = obj.fit_gaussian_process()
    assert explicit is not first and len(calls) == 2
    assert obj.get_default_gaussian_process() is first
    query = first.coordinates[::10]
    mean, covariance = first(query)
    mean2, variance = first(query, return_var=True)
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_allclose(np.diag(covariance), variance, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize('weighting,class_weighting', [('flat','flat'), ('redshift','kaggle')])
def test_complete_lifecycle_numeric_bindings(weighting, class_weighting):
    payload = synthetic_payload(); payload['config'].update(weighting=weighting, class_weighting=class_weighting)
    original = copy.deepcopy(payload)
    expected = predict_runtime(train_runtime(prepare_runtime(payload)))
    calls = {name:0 for name in OPERATIONS}
    def track(name, fn):
        def invoke(*args, **kwargs): calls[name] += 1; return fn(*args, **kwargs)
        return invoke
    operations = {name:track(name,fn) for name,fn in OPERATIONS.items()}
    prepared = prepare(payload, operations)
    # Every successfully augmented object must also use the supplied functions.
    augment = prepared.augmentor.augment_object; augmented = []
    def checked(*args, **kwargs):
        result = augment(*args, **kwargs)
        if result is not None:
            assert isinstance(result, NumericObject)
            augmented.append(result)
        return result
    with patch.object(prepared.augmentor, 'augment_object', side_effect=checked):
        completed = train_decomposed(prepared, reference_stratified_folds, balanced_weights, fit_lightgbm_classifier)
    actual = predict(completed, predict_scores, normalize_scores)
    assert all(calls.values()) and augmented
    assert len(completed.classifier.classifiers) == payload['config']['num_folds']
    assert actual['object_ids'] == expected['object_ids'] and actual['classes'] == expected['classes']
    np.testing.assert_allclose(actual['probabilities'], expected['probabilities'], atol=1e-8, rtol=1e-8)
    assert payload == original

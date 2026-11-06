"""Execute registered numeric bindings through the full synthetic lifecycle."""
import inspect
from pathlib import Path
import numpy as np
import pytest
from sciona.atoms.ml import decomposed_plasticc_numeric as providers
from sciona.atoms.ml import decomposed_plasticc_lifecycle as lifecycle
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.plasticc_runtime import prepare_runtime, train_runtime, predict_runtime
from scripts.validate_plasticc_runtime import synthetic_payload


@pytest.mark.parametrize('module', [providers, lifecycle])
def test_registration_inventory(module):
    directory = Path(inspect.getfile(module)).resolve().parents[4]
    specs = [spec for spec in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
             artifact_root=directory/'src/sciona/atoms') if spec.import_module == module.__name__]
    expected = {'initial_amplitude', 'fit_gp', 'scale_covariance', 'observation_variance', 'condition',
                'predict_gp', 'balanced_weights', 'reference_stratified_folds', 'normalize_scores',
                'predict_scores', 'average_folds'}
    if module is lifecycle:
        expected = {'prepare', 'augment', 'featurizer', 'extract_features', 'assign_folds', 'training_weights',
                    'training_inputs', 'fit_fold', 'fit_folds', 'complete', 'prediction_inputs', 'task_policy', 'encode'}
    assert {spec.source_symbol for spec in specs} == expected
    assert len({spec.version_id for spec in specs}) == len(expected)
    for spec in specs:
        fn = getattr(module, spec.source_symbol)
        assert Path(inspect.getfile(inspect.unwrap(fn))).resolve() == spec.file_path.resolve()
        assert all(p.annotation is not inspect.Parameter.empty for p in inspect.signature(fn).parameters.values())
    with pytest.raises(ValueError):
        module.concrete_only(object())


@pytest.mark.parametrize('weighting,class_weighting', [('flat', 'flat'), ('redshift', 'kaggle')])
def test_registered_numeric_lifecycle(weighting, class_weighting):
    payload = synthetic_payload()
    payload['config'].update(weighting=weighting, class_weighting=class_weighting)
    expected = predict_runtime(train_runtime(prepare_runtime(payload)))
    prepared = lifecycle.prepare(payload, providers.initial_amplitude, providers.fit_gp, matern_kernel_matrix,
        providers.scale_covariance, providers.observation_variance, providers.condition, providers.predict_gp)
    population = lifecycle.augment(prepared)
    policy = lifecycle.featurizer(prepared)
    features = lifecycle.extract_features(population, policy)
    folds = lifecycle.assign_folds(population, prepared, providers.reference_stratified_folds)
    weights, class_weights = lifecycle.training_weights(population, prepared, providers.balanced_weights)
    labels, parameters = lifecycle.training_inputs(population, prepared)
    result = lifecycle.fit_folds(features, labels, weights, folds, parameters, lifecycle.fit_fold)
    completed = lifecycle.complete(prepared, population, policy, result, weights, class_weights)
    query, policy, models, classes = lifecycle.prediction_inputs(completed)
    features = lifecycle.extract_features(query, policy)
    averaged = providers.average_folds(models, features, classes, providers.predict_scores, providers.normalize_scores)
    actual = lifecycle.encode(lifecycle.task_policy(query, averaged))
    assert actual['classes'] == expected['classes'] and actual['object_ids'] == expected['object_ids']
    np.testing.assert_allclose(actual['probabilities'], expected['probabilities'], atol=1e-8, rtol=1e-8)

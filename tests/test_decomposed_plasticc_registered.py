"""Execute registered numeric bindings through the full synthetic lifecycle."""
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sciona.atoms.ml import decomposed_plasticc_numeric as providers
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_plasticc_observations import prepare
from sciona.decomposed_plasticc_training import train
from sciona.decomposed_plasticc_prediction import predict
from sciona.plasticc_runtime import prepare_runtime, train_runtime, predict_runtime
from sciona.plasticc_training import fit_lightgbm_classifier
from scripts.validate_plasticc_runtime import synthetic_payload


def test_registration_inventory():
    directory = Path(inspect.getfile(providers)).resolve().parents[4]
    specs = [spec for spec in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
             artifact_root=directory/'src/sciona/atoms') if spec.import_module == providers.__name__]
    expected = {'initial_amplitude', 'fit_gp', 'scale_covariance', 'observation_variance', 'condition',
                'predict_gp', 'balanced_weights', 'reference_stratified_folds', 'normalize_scores',
                'predict_scores', 'average_folds'}
    assert {spec.source_symbol for spec in specs} == expected
    assert len({spec.version_id for spec in specs}) == len(expected)
    for spec in specs:
        fn = getattr(providers, spec.source_symbol)
        assert Path(inspect.getfile(inspect.unwrap(fn))).resolve() == spec.file_path.resolve()
        assert all(p.annotation is not inspect.Parameter.empty for p in inspect.signature(fn).parameters.values())
    with pytest.raises(ValueError):
        providers.concrete_only(object())


@pytest.mark.parametrize('weighting,class_weighting', [('flat', 'flat'), ('redshift', 'kaggle')])
def test_registered_numeric_lifecycle(weighting, class_weighting):
    payload = synthetic_payload()
    payload['config'].update(weighting=weighting, class_weighting=class_weighting)
    expected = predict_runtime(train_runtime(prepare_runtime(payload)))
    operations = dict(amplitude=providers.initial_amplitude, fit=providers.fit_gp, kernel=matern_kernel_matrix,
        scale=providers.scale_covariance, noise=providers.observation_variance,
        condition=providers.condition, predict=providers.predict_gp)
    completed = train(prepare(payload, operations), providers.reference_stratified_folds,
                      providers.balanced_weights, fit_lightgbm_classifier)
    actual = predict(completed, providers.predict_scores, providers.normalize_scores)
    averaged = providers.average_folds(completed.classifier.classifiers, completed.prediction.features,
        completed.classifier.train_predictions.columns, providers.predict_scores, providers.normalize_scores)
    pd.testing.assert_frame_equal(averaged, completed.prediction.predictions, check_exact=True)
    assert actual['classes'] == expected['classes'] and actual['object_ids'] == expected['object_ids']
    np.testing.assert_allclose(actual['probabilities'], expected['probabilities'], atol=1e-8, rtol=1e-8)

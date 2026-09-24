"""Synthetic native-model roundtrips and contract rejection."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sciona.atoms.ml.xgboost.model_io import (fit_regression_model, fit_binary_model,
    predict_regression_model, predict_binary_model)


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(1291)
    x = rng.normal(size=(96, 3))
    y = 2*x[:, 0]-x[:, 1]+rng.normal(0, .1, 96)
    labels = (y > np.median(y)).astype(np.int64)
    names = ['feature_b', 'feature_a', 'feature_c']
    return x, y, labels, names, fit_regression_model(x, y, names), fit_binary_model(x, labels, names)


@pytest.mark.parametrize('domain,kind', [('manufacturing_dimensions', 'regression'), ('energy_events', 'binary')])
def test_native_predictions_match_sklearn_backend_and_json_roundtrip(fitted, domain, kind):
    x, y, labels, names, regression, binary = fitted
    assert domain
    estimator = (xgb.XGBRegressor if kind == 'regression' else xgb.XGBClassifier)(n_jobs=1, device='cpu')
    estimator.fit(pd.DataFrame(x, columns=names), y if kind == 'regression' else labels)
    frame = pd.DataFrame(x[:12], columns=names)
    expected = estimator.predict(frame) if kind == 'regression' else estimator.predict_proba(frame)[:, 1]
    state = regression if kind == 'regression' else binary
    predict = predict_regression_model if kind == 'regression' else predict_binary_model
    np.testing.assert_array_equal(predict(x[:12], names, json.loads(json.dumps(state))), expected)


def test_native_models_load_in_fresh_process(fitted):
    x, _, _, names, regression, binary = fitted
    payload = dict(x=x[:12].tolist(), names=names, regression=regression, binary=binary)
    code = '''import json,sys,numpy as np
from sciona.atoms.ml.xgboost.model_io import predict_regression_model,predict_binary_model
p=json.load(sys.stdin);x=np.asarray(p['x'],dtype=np.float64)
print(json.dumps([predict_regression_model(x,p['names'],p['regression']).tolist(),predict_binary_model(x,p['names'],p['binary']).tolist()]))
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    child = subprocess.run([sys.executable, '-c', code], input=json.dumps(payload), text=True,
                           capture_output=True, check=True, timeout=60, env=env)
    reg, binary_result = json.loads(child.stdout)
    np.testing.assert_array_equal(reg, predict_regression_model(x[:12], names, regression))
    np.testing.assert_array_equal(binary_result, predict_binary_model(x[:12], names, binary))


@pytest.mark.parametrize('change', ['order', 'version', 'task', 'payload', 'missing'])
def test_tampered_model_state_rejected(fitted, change):
    x, _, _, names, regression, _ = fitted
    state = copy.deepcopy(regression)
    if change == 'order': state['feature_names'] = names[::-1]
    elif change == 'version': state['backend_version'] = '0.0'
    elif change == 'task': state['task'] = 'binary'
    elif change == 'payload': state['payload'] = state['payload'][:-4]+'AAAA'
    else: del state['payload_sha256']
    with pytest.raises(ValueError):
        predict_regression_model(x[:1], names, state)


def test_wrong_prediction_order_rejected(fitted):
    x, _, _, names, regression, _ = fitted
    with pytest.raises(ValueError, match='feature order'):
        predict_regression_model(x[:1, ::-1], names[::-1], regression)


def test_missing_features_follow_backend_but_infinities_fail():
    x = np.arange(40, dtype=np.float64).reshape(20, 2)
    x[::2, 0] = np.nan
    y = np.arange(20, dtype=np.float64)
    state = fit_regression_model(x, y, ['a', 'b'])
    assert np.isfinite(predict_regression_model(x, ['a', 'b'], state)).all()
    x[0, 0] = np.inf
    with pytest.raises(ValueError): fit_regression_model(x, y, ['a', 'b'])


def test_invalid_targets_and_names_rejected(fitted):
    x, y, _, names, _, _ = fitted
    with pytest.raises(ValueError): fit_binary_model(x, np.zeros(len(x), dtype=np.int64), names)
    with pytest.raises(ValueError): fit_regression_model(x, y[:-1], names)
    with pytest.raises(ValueError): fit_regression_model(x, y, ['a', 'a', 'b'])
    with pytest.raises(ValueError): fit_regression_model(x, y, [['a'], 'b', 'c'])


def test_regression_witness_preserves_target_dimension():
    from sciona.ghost.abstract import AbstractArray
    from sciona.ghost.dimensions import DimensionalSignature
    from sciona.atoms.ml.xgboost.model_io import witness_fit_regression_model, witness_predict_regression_model
    names = ['a', 'b']
    dimension = DimensionalSignature(L=1)
    state = witness_fit_regression_model(AbstractArray(shape=(20, 2)),
        AbstractArray(shape=(20,), dim=dimension), names)
    output = witness_predict_regression_model(AbstractArray(shape=(7, 2)), names, state)
    assert output.shape == (7,) and output.dtype == 'float32' and output.dim == dimension
    with pytest.raises(ValueError, match='ordered feature'):
        witness_predict_regression_model(AbstractArray(shape=(7, 2)), names[::-1], state)


def test_binary_witness_rejects_dimensioned_labels():
    from sciona.ghost.abstract import AbstractArray
    from sciona.ghost.dimensions import DimensionalSignature
    from sciona.atoms.ml.xgboost.model_io import witness_fit_binary_model
    with pytest.raises(ValueError, match='dimensionless'):
        witness_fit_binary_model(AbstractArray(shape=(20, 2)),
            AbstractArray(shape=(20,), dtype='int64', dim=DimensionalSignature(T=1)), ['a', 'b'])

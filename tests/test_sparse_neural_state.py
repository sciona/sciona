import copy
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sciona.sparse_neural_regression import SparseRegressor, QuantileSparseRegressor, prelu, tf
from sciona.sparse_neural_state import export_prediction_state, restore_prediction_state


@pytest.mark.parametrize('cls', [SparseRegressor, QuantileSparseRegressor])
@pytest.mark.parametrize('activation', [prelu, tf.nn.relu])
def test_json_prediction_roundtrip_after_training_session_closed(cls, activation):
    X = csr_matrix(np.array([[0., 1.], [1., 0.], [1., 1.], [.5, .5]], dtype=np.float32))
    model = cls(n_hidden=(3, 2), n_epoch=1, batch_size=4, n_bins=2, actfunc=activation)
    try:
        model.fit(X, np.array([1., 2., 3., 2.]))
        expected = model.predict(X)
        state = json.loads(json.dumps(export_prediction_state(model, ['a', 'b'], 'features', 'response'), allow_nan=False))
    finally:
        model.close()
    restored = restore_prediction_state(state, ['a', 'b'], 'features', 'response')
    try:
        np.testing.assert_array_equal(restored.predict(X), expected)
        assert not hasattr(restored, '_train_op')
        assert set(export_prediction_state(restored, ['a', 'b'], 'features', 'response')['weights']) == set(state['weights'])
    finally:
        restored.close()


@pytest.fixture(scope='module')
def state():
    model = SparseRegressor(n_hidden=(2, 2), n_epoch=1, batch_size=2, use_target_scaling=False)
    try:
        model.fit(csr_matrix(np.eye(2, dtype=np.float32)), np.array([1., 2.]))
        return export_prediction_state(model, ['a', 'b'], 'f', 'r')
    finally:
        model.close()


@pytest.mark.parametrize('fault', ['extra', 'missing_weight', 'shape', 'nan_weight', 'decoder', 'activation'])
def test_corrupt_state_rejected(state, fault):
    broken = copy.deepcopy(state)
    first = next(iter(broken['weights']))
    if fault == 'extra': broken['executable'] = 'not permitted'
    elif fault == 'missing_weight': del broken['weights'][first]
    elif fault == 'shape': broken['weights'][first] = [1.]
    elif fault == 'nan_weight': broken['weights'][first][0][0] = float('nan')
    elif fault == 'decoder': broken['decoder'] = {'mean': 0., 'scale': 1.}
    elif fault == 'activation': broken['activation'] = 'arbitrary-code'
    with pytest.raises(ValueError):
        restore_prediction_state(broken, ['a', 'b'], 'f', 'r')


@pytest.mark.parametrize('columns,f,r', [(['b', 'a'], 'f', 'r'), (['a', 'b'], 'other', 'r'), (['a', 'b'], 'f', 'other')])
def test_feature_and_coordinate_identity_rejected(state, columns, f, r):
    with pytest.raises(ValueError, match='identities'):
        restore_prediction_state(state, columns, f, r)


def test_unscaled_mse_state_replay(state):
    restored = restore_prediction_state(state, ['a', 'b'], 'f', 'r')
    try:
        assert np.isfinite(restored.predict(csr_matrix(np.eye(2, dtype=np.float32)))).all()
        assert restored.use_target_scaling is False
    finally:
        restored.close()

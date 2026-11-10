import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sciona.sparse_neural_regression import SparseRegressor, QuantileSparseRegressor


def population():
    return csr_matrix(np.array([[0., 1.], [1., 0.], [1., 1.], [.5, .5]], dtype=np.float32)), np.array([1., 2., 3., 2.])


@pytest.mark.parametrize('cls', [SparseRegressor, QuantileSparseRegressor])
def test_session_lifecycle_and_prediction_boundaries(cls):
    X, y = population()
    model = cls(n_hidden=(3, 2), n_epoch=1, batch_size=3, n_bins=2)
    with pytest.raises(ValueError, match='fitted'):
        model.predict(X)
    try:
        model.fit(X, y)
        assert model.update_count_ == 2
        assert np.isfinite(model.predict(X)).all()
        assert model.predict(csr_matrix((0, 2), dtype=np.float32)).shape == (0,)
        with pytest.raises(ValueError, match='width'):
            model.predict(csr_matrix((1, 3), dtype=np.float32))
    finally:
        model.close()
    assert model._session is None and not model.is_fitted
    with pytest.raises(ValueError):
        model.predict(X)


def test_refit_replaces_quantile_population():
    X, y = population()
    model = QuantileSparseRegressor(n_hidden=(3, 2), n_epoch=1, batch_size=4, n_bins=2)
    try:
        model.fit(X, y)
        old = model.percentiles.copy()
        model.fit(X, y+10)
        np.testing.assert_array_equal(model.percentiles, old+10)
        assert model.update_count_ == 1
    finally:
        model.close()


def test_graph_build_failure_closes_session():
    X, y = population()
    def fail(_):
        raise ValueError('injected activation failure')
    model = SparseRegressor(n_hidden=(3, 2), n_epoch=1, actfunc=fail)
    with pytest.raises(ValueError, match='injected'):
        model.fit(X, y)
    assert model._session is None and not model.is_fitted


@pytest.mark.parametrize('controls', [dict(n_hidden=(2,)), dict(n_epoch=0), dict(batch_size=True),
    dict(seed=-1), dict(learning_rate=float('nan')), dict(quantile_precision=0)])
def test_bad_controls_fail_before_session_creation(controls):
    X, y = population()
    model = SparseRegressor(**controls)
    with pytest.raises(ValueError):
        model.fit(X, y)
    assert not hasattr(model, '_session')


def test_failed_refit_does_not_leave_previous_model_active():
    X, y = population()
    model = SparseRegressor(n_hidden=(3, 2), n_epoch=1, batch_size=4)
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.fit(X, np.full(4, np.nan))
    assert model._session is None and not model.is_fitted

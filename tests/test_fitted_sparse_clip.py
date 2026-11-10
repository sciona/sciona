import pickle

import numpy as np
import pytest
from scipy import sparse

from sciona.fitted_sparse_clip import FittedSparseClip, fit_sparse_clip


def test_frozen_training_bound_and_nonmutation():
    training = sparse.csr_matrix([[0., 2., np.nan], [-1., 5., 0.]])
    query = sparse.csr_matrix([[np.inf, -np.inf, np.nan], [100., 3., -2.]])
    before = query.data.copy()
    state = fit_sparse_clip(training)
    result = state.transform(query)
    np.testing.assert_array_equal(result.toarray(), [[5., 0., 0.], [5., 3., 0.]])
    np.testing.assert_array_equal(query.data, before)
    assert result.dtype == np.float32
    assert state.upper == 5
    assert not np.shares_memory(result.data, query.data)
    replay = pickle.loads(pickle.dumps(state)).transform(query)
    np.testing.assert_array_equal(result.toarray(), replay.toarray())


def test_explicit_zero_and_empty_query():
    training = sparse.csr_matrix(([0.], ([0], [1])), shape=(2, 3))
    state = fit_sparse_clip(training)
    assert state.upper == 0
    assert state.transform(sparse.csr_matrix((0, 3))).shape == (0, 3)
    assert state.transform([[2., -1., 0.]]).sum() == 0


def test_cross_domain_positive_unit_scaling():
    # Synthetic count features and synthetic intensity features in different units.
    training = np.array([[0., 2.], [4., 1.]])
    query = np.array([[8., -1.], [3., 0.]])
    baseline = fit_sparse_clip(training).transform(query).toarray()
    scaled = fit_sparse_clip(training * 10).transform(query * 10).toarray()
    np.testing.assert_array_equal(scaled, baseline * 10)


@pytest.mark.parametrize('training', [np.zeros((2, 3)), [[np.nan]], [[np.inf]],
    [[-np.inf]], [[-2.]], np.zeros((0, 3)), np.zeros((2, 0)), [[1j]], ['bad']])
def test_unsupported_fit_boundaries(training):
    with pytest.raises(ValueError):
        fit_sparse_clip(training)


def test_width_and_duplicate_storage_rejected():
    state = fit_sparse_clip([[1., 2.]])
    with pytest.raises(ValueError, match='width'):
        state.transform([[1.]])
    duplicate = sparse.csr_matrix(([1., 2.], [0, 0], [0, 2]), shape=(1, 2))
    with pytest.raises(ValueError, match='Canonical'):
        fit_sparse_clip(duplicate)


@pytest.mark.parametrize('width,upper', [(0, 1), (True, 1), (1, -1), (1, np.nan), (1, np.inf)])
def test_invalid_restored_state_rejected(width, upper):
    with pytest.raises(ValueError):
        FittedSparseClip(width, upper)

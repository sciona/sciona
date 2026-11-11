import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sciona.sparse_clip_transformer import SparseClipTransformer


def test_frozen_clipping_and_explicit_coordinate():
    train = csr_matrix(np.array([[1., 2.], [3., 0.]], dtype=np.float32))
    model = SparseClipTransformer('synthetic-feature-space').fit(train)
    query = csr_matrix(np.array([[-1., 4.]], dtype=np.float32))
    np.testing.assert_array_equal(model.transform(query).toarray(), [[0., 3.]])
    np.testing.assert_array_equal(query.toarray(), [[-1., 4.]])
    model.feature_coordinate = 'different-feature-space'
    with pytest.raises(ValueError): model.transform(query)


def test_failed_refit_invalidates_and_rejects_dense_input():
    model = SparseClipTransformer('synthetic-feature-space').fit(csr_matrix([[1., 2.]]))
    with pytest.raises(ValueError): model.fit(np.array([[1., 2.]]))
    with pytest.raises(ValueError): model.transform(csr_matrix([[1., 2.]]))

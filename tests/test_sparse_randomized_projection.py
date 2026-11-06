"""Sparse projection arithmetic and full retained text-pair lifecycle checks."""
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sciona.sparse_randomized_projection import fit_projection, transform_projection


@pytest.mark.parametrize('shape',[(9,19),(19,9),(16,16),(40,23)])
@pytest.mark.parametrize('seed',[3,42])
def test_factors_and_projection(shape,seed):
    rng=np.random.default_rng(seed)
    values=rng.normal(size=shape);values[rng.random(shape)<.6]=0
    matrix=sparse.csr_matrix(values);before=matrix.copy()
    # Historical library reference may emit the independently reproduced warnings.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        reference=TruncatedSVD(n_components=2,n_iter=5,n_oversamples=10,random_state=seed).fit(matrix)
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        components,singular=fit_projection(matrix,2,5,10,seed)
        result=transform_projection(matrix,components)
    np.testing.assert_allclose(components,reference.components_,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(singular,reference.singular_values_,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(result,reference.transform(matrix),rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(components@components.T,np.eye(2),atol=1e-12)
    assert (matrix!=before).nnz==0


@pytest.mark.parametrize('args',[(True,5,10,42),(0,5,10,42),(2,0,10,42),(2,5,0,42),(2,5,10,-1)])
def test_invalid_controls(args):
    with pytest.raises(ValueError):fit_projection(sparse.eye(4,format='csr'),*args)


def test_invalid_values_and_transform():
    x=sparse.csr_matrix(np.eye(4));x.data[0]=np.nan
    with pytest.raises(ValueError):fit_projection(x,2,5,10,42)
    with pytest.raises(ValueError):transform_projection(x,np.ones((2,4)))
    with pytest.raises(ValueError):transform_projection(sparse.eye(4,format='csr'),np.ones((2,3)))


def test_retained_lifecycle_with_new_projection(monkeypatch):
    from sciona import text_pair_features
    from tests.test_text_pair_training import test_full_ensemble_calibration_prediction_and_reproducibility, data
    from sciona.text_pair_training import fit_ensemble, calibrate_threshold, predict_pairs
    train,y,cal,cy,query=data()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        baseline=predict_pairs(calibrate_threshold(fit_ensemble(train,y),cal,cy),query)
    class Projection:
        def __init__(self,n_components,random_state):self.rank,self.seed=n_components,random_state
        def fit(self,matrix):
            self.components_,self.singular_values_=fit_projection(matrix,self.rank,5,10,self.seed)
            return self
        def transform(self,matrix):return transform_projection(matrix,self.components_)
    monkeypatch.setattr(text_pair_features,'TruncatedSVD',Projection)
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        test_full_ensemble_calibration_prediction_and_reproducibility()
        current=predict_pairs(calibrate_threshold(fit_ensemble(train,y),cal,cy),query)
    assert current['matches']==baseline['matches']
    assert current['calibration_f1']==baseline['calibration_f1']
    np.testing.assert_allclose(current['scores'],baseline['scores'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(current['threshold'],baseline['threshold'],rtol=1e-10,atol=1e-12)


def test_unsorted_duplicate_storage_preserved():
    x=sparse.csr_matrix((np.array([1.,2.,3.,4.,5.,6.]),np.array([2,0,2,1,2,0]),np.array([0,3,4,6])),shape=(3,3))
    before=(x.data.copy(),x.indices.copy(),x.indptr.copy())
    components,_=fit_projection(x,1,5,10,42)
    projected=transform_projection(x,components)
    np.testing.assert_allclose(projected,np.einsum('ij,kj->ik',x.toarray(),components),atol=1e-12)
    for a,b in zip((x.data,x.indices,x.indptr),before):np.testing.assert_array_equal(a,b)

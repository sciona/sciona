"""Synthetic fitted-state parity, independent scoring and rejection boundaries."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sciona.atoms.ml.sklearn.linear_model.sparse_binary import fit_sparse_binary_logistic as fit, predict_sparse_binary_probability as predict


def sample():
    return csr_matrix(np.array([[-2.,0.],[-1.,1.],[1.,0.],[2.,1.]])),np.array([0,0,1,1],dtype=np.int64)


@pytest.mark.parametrize('seed', [3,12])
def test_sklearn_parity_without_densification_or_mutation(seed,monkeypatch):
    x,y=sample();before=x.copy();labels=y.copy()
    expected=LogisticRegression(max_iter=1000,random_state=seed).fit(x,y)
    def fail(*a,**k):pytest.fail('Unexpected feature densification')
    monkeypatch.setattr(csr_matrix,'toarray',fail)
    actual=fit(x,y,1.,1e-4,1000,seed)
    np.testing.assert_array_equal(actual.coef_,expected.coef_)
    np.testing.assert_array_equal(actual.intercept_,expected.intercept_)
    np.testing.assert_array_equal(predict(x,actual),expected.predict_proba(x)[:,1])
    for attr in ('data','indices','indptr'):np.testing.assert_array_equal(getattr(x,attr),getattr(before,attr))
    np.testing.assert_array_equal(y,labels)


@pytest.mark.parametrize('case', ['dense','float32','empty','nan','noncanonical'])
def test_invalid_features(case):
    x,y=sample()
    if case=='dense':x=x.toarray()
    if case=='float32':x=x.astype(np.float32)
    if case=='empty':x=x[:0]
    if case=='nan':x.data[0]=float('nan')
    if case=='noncanonical':x=csr_matrix((np.ones(2),np.array([0,0]),np.array([0,2,2,2,2])),shape=(4,2))
    with pytest.raises(ValueError):fit(x,y,1.,1e-4,1000,12)


@pytest.mark.parametrize('labels', [np.array([0.,0.,1.,1.]),np.array([0,0,0,0]),np.array([0,0,1,2]),np.array([0,1])])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):fit(sample()[0],labels,1.,1e-4,1000,12)


@pytest.mark.parametrize('controls', [(0.,1e-4,1000,12),(1.,float('inf'),1000,12),(1.,1e-4,True,12),(1.,1e-4,1000,-1),(1.,1e-4,1000,2**32)])
def test_invalid_controls(controls):
    with pytest.raises(ValueError):fit(*sample(),*controls)


def test_nonconvergence_is_rejected():
    with pytest.raises(ConvergenceWarning):fit(*sample(),1.,1e-15,1,12)


def test_independent_known_coefficients_and_no_refit(monkeypatch):
    model=LogisticRegression();model.classes_=np.array([0,1]);model.n_features_in_=2
    model.coef_=np.array([[2.,-1.]]);model.intercept_=np.array([.5])
    def fail(*a,**k):pytest.fail('Prediction must not fit')
    monkeypatch.setattr(model,'fit',fail)
    x,_=sample()
    np.testing.assert_allclose(predict(x,model),expit(np.array([-3.5,-2.5,2.5,3.5])),rtol=0,atol=0)


def test_invalid_prediction_state():
    x,y=sample()
    with pytest.raises(ValueError):predict(x,LogisticRegression())
    model=fit(x,y,1.,1e-4,1000,12)
    with pytest.raises(ValueError):predict(x[:,:1],model)
    model.coef_[0,0]=float('nan')
    with pytest.raises(ValueError):predict(x,model)

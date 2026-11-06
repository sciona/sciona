"""Synthetic sparse vocabulary semantics and one-pass source consumption."""
import copy
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sciona.atoms.ml.sklearn.feature_extraction.sparse_numeric import fit_numeric_vocabulary, transform_numeric_vocabulary


def test_source_parity_and_frozen_unknown_features():
    rows=[{'z':2.,'a':-1.},{},{'a':0.,'b':3.}]
    original=copy.deepcopy(rows);seen=[]
    def records():
        for i,row in enumerate(rows):seen.append(i);yield row
    state,matrix=fit_numeric_vocabulary(records())
    reference=DictVectorizer(sparse=True,sort=True)
    expected=reference.fit_transform(rows)
    assert isinstance(matrix,csr_matrix) and matrix.dtype==np.float64
    assert seen==[0,1,2] and rows==original
    assert state.vocabulary_==reference.vocabulary_
    for attr in ('data','indices','indptr'):
        np.testing.assert_array_equal(getattr(matrix,attr),getattr(expected,attr))
    snapshot=copy.deepcopy(state.__dict__)
    query=[{'unseen':9.,'a':4.},{}]
    actual=transform_numeric_vocabulary(iter(query),state)
    np.testing.assert_array_equal(actual.toarray(),[[4.,0.,0.],[0.,0.,0.]])
    assert state.__dict__==snapshot


@pytest.mark.parametrize('rows', [[],[{}],[{'':1.}],[{1:2.}],[{'x':True}],[{'x':'a'}],[{'x':float('nan')}],[{'x':float('inf')} ]])
def test_invalid_fit_records(rows):
    with pytest.raises(ValueError):fit_numeric_vocabulary(iter(rows))


@pytest.mark.parametrize('change', ['dense','names','unfitted'])
def test_invalid_fitted_state(change):
    state,_=fit_numeric_vocabulary([{'a':1.}])
    if change=='dense':state.sparse=False
    if change=='names':state.feature_names_=['different']
    if change=='unfitted':state=DictVectorizer()
    with pytest.raises(ValueError):transform_numeric_vocabulary([{'a':1.}],state)


def test_transform_validates_unknown_values_and_empty_iterable():
    state,_=fit_numeric_vocabulary([{'a':1.}])
    for rows in ([],[{'unknown':float('nan')}],[{'unknown':True}]):
        with pytest.raises(ValueError):transform_numeric_vocabulary(iter(rows),state)


def test_wide_inputs_stay_sparse(monkeypatch):
    def fail(*args,**kwargs):pytest.fail('Unexpected dense conversion')
    monkeypatch.setattr(csr_matrix,'toarray',fail)
    state,matrix=fit_numeric_vocabulary(({f'feature_{i}':1.} for i in range(1000)))
    assert matrix.shape==(1000,1000) and matrix.nnz==1000
    transformed=transform_numeric_vocabulary(({'feature_3':2.} for _ in range(2)),state)
    assert transformed.shape==(2,1000) and transformed.nnz==2

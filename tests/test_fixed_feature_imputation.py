import numpy as np
import pytest
from sciona.fixed_feature_imputation import fixed_integer_features


def test_query_batch_invariance_and_nonmutation():
    values=np.array([[np.nan,1.75],[91.,np.nan]])
    before=values.copy();fills=np.array([0.,-2.8])
    result=fixed_integer_features(values,fills)
    np.testing.assert_array_equal(result,[[0,1],[91,-2]])
    for row in range(2):np.testing.assert_array_equal(result[row:row+1],fixed_integer_features(values[row:row+1],fills))
    np.testing.assert_array_equal(values,before)
    assert result.dtype==np.int16


@pytest.mark.parametrize('dtype,value',[('int16',40000.),('int16',-40000.),('int64',float(2**63))])
def test_overflow_rejected(dtype,value):
    with pytest.raises(ValueError,match='integer range'):fixed_integer_features(np.array([[value]]),np.array([0.]),dtype)


def test_valid_edges_and_empty_query():
    np.testing.assert_array_equal(fixed_integer_features(np.array([[-32768.9,32767.9]]),np.zeros(2)),[[-32768,32767]])
    assert fixed_integer_features(np.empty((0,2)),np.zeros(2)).shape==(0,2)


@pytest.mark.parametrize('values,fills,dtype',[
    (np.array([[np.inf]]),np.zeros(1),'int16'),
    (np.zeros((1,1)),np.array([np.nan]),'int16'),
    (np.zeros((1,2)),np.zeros(1),'int16'),
    (np.zeros((1,1)),np.zeros(1),'float64'),
])
def test_invalid_contract(values,fills,dtype):
    with pytest.raises(ValueError):fixed_integer_features(values,fills,dtype)


def test_integer_precision_not_silently_lost():
    with pytest.raises(ValueError,match='exact float64'):
        fixed_integer_features(np.array([[2**53+1]],dtype=np.int64),np.array([0]),'int64')

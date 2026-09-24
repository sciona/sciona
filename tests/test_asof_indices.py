import numpy as np
import pytest
from sciona.asof_indices import latest_available_indices as lookup


def test_boundaries_unsorted_and_explicit_ties():
    indices,valid=lookup(np.array([5,-10,20,5],dtype=np.int64),np.array([-11,-10,0,5,21],dtype=np.int64))
    np.testing.assert_array_equal(indices,[-1,1,1,3,2])
    np.testing.assert_array_equal(valid,[False,True,True,True,True])


def test_future_observation_does_not_backfill():
    q=np.array([0,5],dtype=np.int64)
    i,v=lookup(np.array([10,20],dtype=np.int64),q)
    assert np.all(i==-1) and not np.any(v)
    i,v=lookup(np.array([-5,10,20],dtype=np.int64),q)
    assert np.all(i==0) and np.all(v)


def test_empty_and_extreme_clocks():
    q=np.array([np.iinfo(np.int64).min,np.iinfo(np.int64).max],dtype=np.int64)
    i,v=lookup(q,q)
    np.testing.assert_array_equal(i,[0,1])
    assert v.all()
    i,v=lookup(np.array([],dtype=np.int64),q)
    assert np.all(i==-1) and not v.any()


@pytest.mark.parametrize('times',[np.array([1.]),np.array([[1]],dtype=np.int64)])
def test_invalid_contract(times):
    with pytest.raises(ValueError):lookup(times,np.array([0],dtype=np.int64))

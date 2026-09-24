import numpy as np
import pytest
from sciona.event_delay_windows import event_delay_windows as compute


def arrays():
    return [np.array(['a','b']),np.array([-15,-10],dtype=np.int64),np.array([-20,-11],dtype=np.int64),
        np.array(['a','b']),np.array([1,-30],dtype=np.int64),np.array([-35,-16],dtype=np.int64),np.array([True,True]),
        np.array([0,15],dtype=np.int64),np.array([60],dtype=np.int64),15,1]


def test_future_estimate_changes_only_later_query():
    result=compute(*arrays())
    np.testing.assert_array_equal(result[0],[[2],[2]])
    np.testing.assert_array_equal(result[1],[[1],[2]])
    np.testing.assert_array_equal(result[2],[[5],[10]])
    np.testing.assert_array_equal(result[3],[[5],[15]])


def test_future_actual_event_excluded_and_nonmutation():
    args=arrays();args[2][1]=20
    before=[a.copy() if isinstance(a,np.ndarray) else a for a in args]
    result=compute(*args)
    np.testing.assert_array_equal(result[0],[[1],[1]])
    for old,new in zip(before,args):
        if isinstance(old,np.ndarray):np.testing.assert_array_equal(old,new)


def test_missing_estimates_keep_event_counts():
    args=arrays();args[6][:]=False
    c,v,m,x=compute(*args)
    assert np.all(c==2) and np.all(v==0) and np.isnan(m).all() and np.isnan(x).all()


@pytest.mark.parametrize('slot,value',[(9,0),(10,False),(8,np.array([0],dtype=np.int64)),(6,np.array([True]))])
def test_invalid_contract(slot,value):
    args=arrays();args[slot]=value
    with pytest.raises(ValueError):compute(*args)

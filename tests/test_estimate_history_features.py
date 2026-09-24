import numpy as np
import pytest
from sciona.estimate_history_features import estimate_history_features as compute


def inputs():
    return [np.array(['a','a','b','b']),np.array([-30,-15,-30,-15],dtype=np.int64),
        np.array([60,70,180,150],dtype=np.int64),np.ones(4,dtype=bool),
        np.array(['a','b']),np.array([0,0],dtype=np.int64),np.array([60,120],dtype=np.int64),15,1]


def test_independent_reference_values():
    r=compute(*inputs())
    np.testing.assert_array_equal(r['remaining'],[70,150])
    np.testing.assert_array_equal(r['change_from_first'],[10,-30])
    np.testing.assert_array_equal(r['sum'],[[10,10],[-30,-30]])
    np.testing.assert_array_equal(r['max'],[[10,10],[0,0]])
    np.testing.assert_allclose(r['std'],[[np.std([0,10,0],ddof=1)]*2,[np.std([0,-30,0],ddof=1)]*2])


def test_query_population_order_and_duplicates_are_invariant():
    args=inputs();expected=compute(*args)
    args[4]=np.array(['a','b','a','a']);args[5]=np.array([-5,0,0,0],dtype=np.int64)
    actual=compute(*args)
    for key in expected:
        np.testing.assert_array_equal(actual[key][[2,1]],expected[key])
        np.testing.assert_array_equal(actual[key][2],actual[key][3])


def test_future_and_other_entity_changes_do_not_leak():
    args=inputs();expected=compute(*args)
    args[2][2:]+=100
    for slot,extra in [(0,['a']),(1,[1]),(2,[1000]),(3,[True])]:
        args[slot]=np.concatenate([args[slot],np.array(extra,dtype=args[slot].dtype)])
    actual=compute(*args)
    for key in expected:np.testing.assert_array_equal(actual[key][0],expected[key][0])


def test_rounded_availability_and_last_nonmissing_estimate():
    args=inputs();args[:4]=[np.array(['a','a']),np.array([1,2],dtype=np.int64),np.array([60,999],dtype=np.int64),np.array([True,False])]
    args[4:6]=[np.array(['a','a']),np.array([5,15],dtype=np.int64)]
    r=compute(*args)
    np.testing.assert_array_equal(r['estimate_available'],[False,True])
    np.testing.assert_allclose(r['remaining'],[np.nan,45],equal_nan=True)


def test_synthetic_manufacturing_completion_estimate_reuse():
    args=inputs();args[0]=np.array(['machine_a','machine_a','machine_b','machine_b']);args[4]=np.array(['machine_a','machine_b'])
    # Scale minutes to seconds while preserving all output features in minutes.
    for slot in (1,2,5,6):args[slot]*=60
    args[7]=900;args[8]=60
    expected=compute(*inputs());actual=compute(*args)
    for key in expected:np.testing.assert_allclose(actual[key],expected[key],equal_nan=True)


@pytest.mark.parametrize('slot,value',[(7,0),(8,False),(6,np.array([0],dtype=np.int64)),(3,np.ones(4,dtype=int))])
def test_invalid_contract(slot,value):
    args=inputs();args[slot]=value
    with pytest.raises(ValueError):compute(*args)

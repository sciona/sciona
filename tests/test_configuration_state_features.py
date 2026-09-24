import numpy as np
import pytest
from sciona.configuration_state_features import configuration_state_features as compute


def args():
    return [np.array([10,20],dtype=np.int64),np.array([['07R'],['18L']]),np.arange(0,31,5,dtype=np.int64),['07','18'],['R','L','C'],np.array([2],dtype=np.int64),5]


def test_availability_windows_and_future_perturbation():
    inputs=args();r=compute(*inputs)
    np.testing.assert_array_equal(r['state_valid'],[False,False,True,True,True,True,True])
    np.testing.assert_array_equal(r['changes'][:,0],[False,False,False,False,True,False,False])
    np.testing.assert_array_equal(r['rolling_changes'][:,0,0],[0,0,0,0,1,1,0])
    inputs[1][1,0]='07C';changed=compute(*inputs)
    for key in r:np.testing.assert_array_equal(r[key][:4],changed[key][:4])


def test_literal_substrings_and_suffix_priority():
    inputs=args();inputs[1]=inputs[1].astype(object);inputs[1][0,0]='107R'
    assert compute(*inputs)['categories'][2,0,0]==1


def test_reuse_with_machine_modes_and_prior_state():
    result=compute(np.array([-5,10],dtype=np.int64),
        np.array([['pumpON,valveOFF'],['pumpOFF,valveON']]),
        np.arange(0,21,5,dtype=np.int64),['pump','valve'],['ON','OFF'],
        np.array([2],dtype=np.int64),5)
    np.testing.assert_array_equal(result['categories'][:,0],[[1,2],[1,2],[2,1],[2,1],[2,1]])
    np.testing.assert_array_equal(result['item_counts'][:,0],2)
    np.testing.assert_array_equal(result['changes'][:,0],[False,False,True,False,False])
    np.testing.assert_array_equal(result['window_valid'][:,0],[False,True,True,True,True])


def test_empty_history_is_explicitly_unavailable():
    inputs=args();inputs[0]=np.array([],dtype=np.int64);inputs[1]=np.empty((0,1),dtype=str)
    result=compute(*inputs)
    assert not result['state_valid'].any()
    assert not result['window_valid'].any()
    assert not result['changes'].any()


def test_unsorted_history_ties_select_last_input_without_mutation():
    inputs=args()
    inputs[0]=np.array([20,10,10],dtype=np.int64)
    inputs[1]=np.array([['18L'],['07C'],['07R']])
    before=[value.copy() if isinstance(value,np.ndarray) else value[:] if isinstance(value,list) else value for value in inputs]
    result=compute(*inputs)
    np.testing.assert_array_equal(result['categories'][2,0],[1,0])
    for value,original in zip(inputs,before):np.testing.assert_array_equal(value,original)


@pytest.mark.parametrize('slot,value',[(6,0),(2,np.array([0,7],dtype=np.int64)),(5,np.array([0],dtype=np.int64)),(3,['07','07'])])
def test_invalid_contract(slot,value):
    inputs=args();inputs[slot]=value
    with pytest.raises(ValueError):compute(*inputs)

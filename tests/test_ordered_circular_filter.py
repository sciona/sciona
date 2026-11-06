"""Synthetic boundary and arithmetic contracts for reusable circular filtering."""
import numpy as np
import pytest
from sciona.atoms.numerical.spatial.circular_filter import ordered_circular_filter, witness_ordered_circular_filter


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('axis',[0,1,2])
def test_explicit_axis_signed_weights_order_and_input_preservation(dtype,axis):
    values=np.arange(24,dtype=dtype).reshape(2,3,4)
    before=values.copy()
    result=ordered_circular_filter(values,[0,1,-1,1],[1,.8,-.4,.2],axis)
    expected=((values+np.roll(values,1,axis)*.8)+np.roll(values,-1,axis)*-.4)+np.roll(values,1,axis)*.2
    np.testing.assert_array_equal(result,expected)
    np.testing.assert_array_equal(values,before)
    assert result.dtype==dtype and not np.shares_memory(values,result)


def test_left_to_right_rounding_is_retained():
    values=np.ones(3,dtype=np.float32)
    result=ordered_circular_filter(values,[0,0,0],[1e8,-1e8,1],0)
    np.testing.assert_array_equal(result,np.ones(3,dtype=np.float32))


@pytest.mark.parametrize('values,shifts,weights,axis',[
    (np.ones(3,dtype=int),[0],[1],0),
    (np.array([np.nan]),[0],[1],0),
    (np.array([],dtype=float),[0],[1],0),
    (np.ones(3),[],[],0),
    (np.ones(3),[.5],[1],0),
    (np.ones(3),[True],[1],0),
    (np.ones(3),[0],[True],0),
    (np.ones(3),[0],[np.inf],0),
    (np.ones(3),[0],[1,2],0),
    (np.ones(3),[0],[1],-1),
    (np.ones(3),[0],[1],True),
    (np.ones(3),[0],[1],1),
    (np.ones(3,dtype=np.float32),[0],[1e100],0),
    (np.full(3,np.finfo(np.float32).max,dtype=np.float32),[0,0],[1,1],0),
])
def test_invalid_controls_and_nonfinite_arithmetic_rejected(values,shifts,weights,axis):
    with pytest.raises(ValueError):ordered_circular_filter(values,shifts,weights,axis)


def test_abstract_execution_is_not_claimed():
    with pytest.raises(ValueError,match='Concrete'):witness_ordered_circular_filter()

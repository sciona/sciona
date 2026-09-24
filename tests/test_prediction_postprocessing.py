import numpy as np
import pytest
from sciona.prediction_postprocessing import clip_truncate_prediction


def test_order_and_negative_truncation():
    values=np.array([-10.,-2.8,-.8,.8,2.8,10.])
    offset=np.array([20.,.5,0.,0.,-.5,-20.])
    before=values.copy()
    np.testing.assert_array_equal(clip_truncate_prediction(values,offset,-3,3),[3,-2,0,0,2,-3])
    np.testing.assert_array_equal(values,before)


@pytest.mark.parametrize('scale', [1,10])
def test_synthetic_unit_change(scale):
    # Constructed inventory and energy estimates share the same numerical rule.
    values=np.array([1.25,2.75,7.5])*scale
    offset=np.array([.25,-.25,1.5])*scale
    result=clip_truncate_prediction(values,offset,0,8*scale)
    np.testing.assert_array_equal(result,np.array([1.5,2.5,8.])*scale if scale==10 else [1,2,8])


@pytest.mark.parametrize('values,offset,lower,upper',[
    (np.array([np.nan]),np.zeros(1),0,3),
    (np.array([1j]),np.zeros(1),0,3),
    (np.ones(2),np.zeros(1),0,3),
    (np.array([1e308]),np.array([1e308]),0,3),
    (np.ones(1),np.zeros(1),3,0),
    (np.ones(1),np.zeros(1),0,2**53),
])
def test_invalid_contract(values,offset,lower,upper):
    with pytest.raises(ValueError):clip_truncate_prediction(values,offset,lower,upper)

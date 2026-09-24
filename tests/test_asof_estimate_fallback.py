import numpy as np
import pytest
from sciona.asof_estimate_fallback import asof_estimate_fallback as compute


def args():
    return [np.array(['a','a','a','b']),np.array([-10,10,10,-10],dtype=np.int64),
        np.array([75,90,105,3000],dtype=np.int64),np.array([True,True,True,True]),
        np.array(['a','a','b','missing','a']),np.array([0,10,0,0,0],dtype=np.int64)]


def policy():return dict(ticks_per_unit=1,offset=30,lower=1,upper=299,missing=15)


def test_availability_ties_clipping_and_duplicate_queries():
    np.testing.assert_array_equal(compute(*args(),**policy()),[45,65,299,15,45])


def test_null_latest_and_future_perturbation():
    inputs=args();expected=compute(*inputs,**policy())
    inputs[2][1:3]=99999
    np.testing.assert_array_equal(compute(*inputs,**policy())[[0,2,3,4]],expected[[0,2,3,4]])
    inputs=args();inputs[3][1:3]=False
    assert compute(*inputs,**policy())[1]==35


def test_cross_domain_fractional_completion_estimates():
    result=compute(np.array(['machine']),np.array([0],dtype=np.int64),np.array([453],dtype=np.int64),
        np.array([True]),np.array(['machine']),np.array([0],dtype=np.int64),
        ticks_per_unit=10,offset=0,lower=0,upper=100,missing=50)
    np.testing.assert_array_equal(result,[45.3])


def test_empty_history():
    inputs=args();inputs[:4]=[value[:0] for value in inputs[:4]]
    np.testing.assert_array_equal(compute(*inputs,**policy()),15)


@pytest.mark.parametrize('name,value',[('ticks_per_unit',0),('missing',500),('offset',np.inf)])
def test_invalid_policy(name,value):
    options=policy();options[name]=value
    with pytest.raises(ValueError):compute(*args(),**options)

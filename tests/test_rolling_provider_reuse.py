"""Independent synthetic numerical and boundary checks for existing rolling stats."""
import math
import icontract
import numpy as np
import pytest
from sciona.atoms.ml.tabular.gradient_boosting.atoms import rolling_statistics


@pytest.mark.parametrize('length',[1,8,33])
@pytest.mark.parametrize('width',[1,2,65])
@pytest.mark.parametrize('strided',[False,True])
def test_all_operations_against_scalar_oracle(length,width,strided):
    rng=np.random.default_rng(length+width)
    values=rng.normal(size=length*(2 if strided else 1)).astype(np.float64)
    if strided:values=values[::2]
    before=values.copy()
    expected=[]
    for end in range(1,length+1):
        window=list(map(float,values[max(0,end-width):end]));mean=math.fsum(window)/len(window)
        std=math.sqrt(math.fsum((v-mean)**2 for v in window)/len(window))
        expected.append([mean,std,min(window),max(window),math.fsum(window)])
    result=rolling_statistics(values,width,['mean','std','min','max','sum'])
    assert result.dtype==np.float64 and result.shape==(length,5)
    np.testing.assert_allclose(result,expected,rtol=1e-13,atol=1e-14)
    assert np.array_equal(values,before)


def test_requested_order_and_duplicates():
    values=np.array([1.,-3.,7.])
    result=rolling_statistics(values,2,['sum','mean','sum'])
    assert np.array_equal(result,[[1.,1.,1.],[-2.,-1.,-2.],[4.,2.,4.]])


@pytest.mark.parametrize('fault',['nan','inf','rank','zero_width','negative_width','empty_ops','unknown_op'])
def test_rejected_inputs(fault):
    values=np.array([1.,2.]);width=2;operations=['mean']
    if fault=='nan':values[0]=np.nan
    elif fault=='inf':values[0]=np.inf
    elif fault=='rank':values=values.reshape(1,2)
    elif fault=='zero_width':width=0
    elif fault=='negative_width':width=-1
    elif fault=='empty_ops':operations=[]
    else:operations=['median']
    with pytest.raises(icontract.ViolationError):rolling_statistics(values,width,operations)


def test_nonfinite_aggregate_rejected():
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(icontract.ViolationError):
            rolling_statistics(np.array([1e308,1e308]),2,['sum'])


@pytest.mark.parametrize('case',['float_width','bool_width','empty_values'])
def test_runtime_permissiveness_is_explicit(case):
    # Supported publication inputs are stricter than these accepted inputs.
    values=np.array([1.,3.,7.])
    if case=='float_width':
        assert np.array_equal(rolling_statistics(values,2.9,['mean']),rolling_statistics(values,2,['mean']))
    elif case=='bool_width':
        assert np.array_equal(rolling_statistics(values,True,['mean'])[:,0],values)
    else:assert rolling_statistics(np.array([],dtype=float),2,['mean']).shape==(0,1)


def test_future_observations_do_not_change_prefix():
    values=np.arange(12,dtype=np.float64)
    before=rolling_statistics(values,4,['mean','std','min','max','sum'])
    values[8:]+=10000
    after=rolling_statistics(values,4,['mean','std','min','max','sum'])
    assert np.array_equal(before[:8],after[:8])
    assert not np.array_equal(before[8:],after[8:])

import numpy as np
import pytest
from sciona.fixed_window_statistics import fixed_window_statistics as compute


def test_boundaries_separate_availability_and_missing_values():
    times=np.array([-60,-30,-20,-10,0],dtype=np.int64)
    event_available=np.array([-60,-30,1,-10,0],dtype=np.int64)
    value_available=np.array([-60,-30,1,1,0],dtype=np.int64)
    values=np.array([100.,np.nan,200.,30.,40.])
    counts,valid,mean,maximum=compute(times,event_available,value_available,values,np.array([0],dtype=np.int64),np.array([60],dtype=np.int64))
    assert (counts.item(),valid.item(),mean.item(),maximum.item())==(3,1,40.,40.)


def test_extreme_time_subtraction_does_not_wrap():
    low=np.iinfo(np.int64).min
    times=np.array([low],dtype=np.int64)
    c,v,m,x=compute(times,times,times,np.array([2.]),times,np.array([60],dtype=np.int64))
    assert (c.item(),v.item(),m.item(),x.item())==(1,1,2.,2.)


def test_empty_values_have_counts_without_fabricated_statistics():
    times=np.array([0],dtype=np.int64)
    c,v,m,x=compute(times,times,times,np.array([np.nan]),times,np.array([1],dtype=np.int64))
    assert c.item()==1 and v.item()==0 and np.isnan(m.item()) and np.isnan(x.item())


def test_unavailable_value_perturbation_and_input_order_invariance():
    times=np.array([-2,-1,0],dtype=np.int64);available=np.array([-2,5,0],dtype=np.int64)
    values=np.array([2.,999.,4.]);query=np.array([0],dtype=np.int64);width=np.array([3],dtype=np.int64)
    baseline=compute(times,times,available,values,query,width)
    values[1]=-999.
    changed=compute(times[::-1],times[::-1],available[::-1],values[::-1],query,width)
    for first,second in zip(baseline,changed):np.testing.assert_array_equal(first,second)


@pytest.mark.parametrize('kind',['bad_width','unaligned','infinite','float_time'])
def test_invalid_contract(kind):
    times=np.array([0],dtype=np.int64);values=np.array([1.]);width=np.array([1],dtype=np.int64)
    event=times
    if kind=='bad_width':width=np.array([0],dtype=np.int64)
    if kind=='unaligned':event=np.array([],dtype=np.int64)
    if kind=='infinite':values=np.array([np.inf])
    if kind=='float_time':times=times.astype(float)
    with pytest.raises(ValueError):compute(times,event,event,values,np.array([0],dtype=np.int64),width)

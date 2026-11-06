"""Synthetic direct-form recurrence oracle for the qualified existing filter."""
import numpy as np
import pytest
from icontract.errors import ViolationError
from sciona.atoms.scipy.signal import lfilter


def recurrence(b,a,x,zi):
    size=max(len(a),len(b));den=np.pad(a/a[0],(0,size-len(a)));num=np.pad(b/a[0],(0,size-len(b)))
    state=np.zeros(size-1) if zi is None else zi.copy();out=[]
    for value in x:
        y=num[0]*value+(state[0] if state.size else 0.)
        previous=state.copy()
        for j in range(size-1):
            state[j]=(previous[j+1] if j+1<len(previous) else 0.)+num[j+1]*value-den[j+1]*y
        out.append(y)
    return np.asarray(out)


@pytest.mark.parametrize('coefficients', [([.25,.5,.25],[1.]),([.3,-.1],[1.,-.4]),([1.,.2],[2.,-.3,.1])])
@pytest.mark.parametrize('initial',[False,True])
@pytest.mark.parametrize('axis',[-1,0])
@pytest.mark.parametrize('size',[1,9,127])
def test_independent_recurrence_and_input_preservation(coefficients,initial,axis,size):
    b,a=[np.asarray(v,dtype=np.float64) for v in coefficients]
    x=np.sin(np.arange(size)*.37)-.2
    zi=np.linspace(-.7,.4,max(len(a),len(b))-1) if initial else None
    saved=[v.copy() if v is not None else None for v in (b,a,x,zi)]
    result=lfilter(b,a,x,axis=axis,zi=zi)
    np.testing.assert_allclose(result,recurrence(b,a,x,zi),rtol=2e-14,atol=2e-14)
    assert isinstance(result,np.ndarray) and result.shape==x.shape and result.dtype==np.float64
    for value,old in zip((b,a,x,zi),saved):
        if value is not None:np.testing.assert_array_equal(value,old)


@pytest.mark.parametrize('b,a,x', [([[1.]],[1.],[1.]),([1.],[[1.]],[1.]),([1.],[0.],[1.]),([1.],[1.],[])])
def test_registered_invalid_inputs_rejected(b,a,x):
    with pytest.raises(ViolationError):lfilter(np.asarray(b),np.asarray(a),np.asarray(x))

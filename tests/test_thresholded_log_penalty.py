import numpy as np
import pytest
import torch
from sciona.atoms.dl.loss.thresholded_log import thresholded_log_penalty
from sciona.dsb_losses import classifier_loss


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('threshold',[.03,.4,1.])
def test_exact_source_values_and_gradients(dtype,threshold):
    p=torch.tensor([[0.,.02,.03],[.2,.5,1.]],dtype=dtype,requires_grad=True)
    w=torch.tensor([[1.,.5,1.],[0.,1.,1.]],dtype=dtype,requires_grad=True)
    rp=p.detach().clone().requires_grad_();rw=w.detach().clone().requires_grad_()
    expected=classifier_loss(torch.tensor([.2,.7],dtype=dtype),rp,torch.tensor([0.,1.],dtype=dtype),rw,miss_threshold=threshold)['miss']
    actual=thresholded_log_penalty(p,w,threshold,.001)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for a,b in zip(torch.autograd.grad(actual,(p,w)),torch.autograd.grad(expected,(rp,rw))):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_explicit_controls_and_independent_derivatives():
    p=torch.tensor([[.1,.5]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([[.25,.75]],dtype=torch.float64,requires_grad=True)
    result=thresholded_log_penalty(p,w,.5,.1)
    torch.testing.assert_close(result,torch.tensor(-.25*np.log(.2)/2,dtype=torch.float64))
    gp,gw=torch.autograd.grad(result,(p,w))
    torch.testing.assert_close(gp,torch.tensor([[-.625,0.]],dtype=torch.float64))
    torch.testing.assert_close(gw,torch.tensor([[-np.log(.2)/2,0.]],dtype=torch.float64))
    # Strict threshold is nondifferentiable at equality; gradcheck stays away from it.
    assert torch.autograd.gradcheck(lambda a,b:thresholded_log_penalty(a,b,.4,.1),(p,w))


@pytest.mark.parametrize('threshold,offset',[(0,.1),(1.1,.1),(float('nan'),.1),(.5,0),(.5,float('inf')),(True,.1)])
def test_bad_controls(threshold,offset):
    with pytest.raises(ValueError):thresholded_log_penalty(torch.ones(1,2),torch.ones(1,2),threshold,offset)


def test_bad_arrays_and_unrepresentable_offset():
    for p,w in [(torch.empty(1,0),torch.empty(1,0)),(torch.tensor([[1.1]]),torch.ones(1,1)),
                (torch.ones(1,1),torch.tensor([[-1.]])),(torch.ones(1,1),torch.ones(1,2)),
                (torch.ones(1,1),torch.ones(1,1,dtype=torch.float64)),
                (torch.ones(1,1,dtype=torch.float16),torch.ones(1,1,dtype=torch.float16))]:
        with pytest.raises(ValueError):thresholded_log_penalty(p,w,.03,.001)
    with pytest.raises(ValueError):thresholded_log_penalty(torch.zeros(1,1),torch.ones(1,1),.5,1e-100)
    with pytest.raises(ValueError):thresholded_log_penalty(torch.zeros(1,1),torch.ones(1,1),.5,1e100)

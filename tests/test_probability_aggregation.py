import numpy as np
import pytest
import torch
from sciona.atoms.dl.probability_aggregation import noisy_or_with_baseline,witness_noisy_or_with_baseline
from sciona.dsb_losses import LearnedNoisyOR
from sciona.ghost.abstract import AbstractArray


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('baseline',[-30.,0.,2.])
@pytest.mark.parametrize('shape',[(1,1),(2,3),(3,5)])
def test_outputs_and_gradients_match_retained_source_operation(dtype,baseline,shape):
    p=torch.linspace(.05,.85,int(np.prod(shape)),dtype=dtype).reshape(shape).requires_grad_()
    b=torch.tensor([baseline],dtype=dtype,requires_grad=True)
    source=LearnedNoisyOR().to(dtype)
    with torch.no_grad():source.baseline.copy_(b)
    reference_p=p.detach().clone().requires_grad_()
    expected=source(reference_p)
    actual=noisy_or_with_baseline(p,b)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    gradients=torch.autograd.grad(actual.sum(),(p,b))
    reference_gradients=torch.autograd.grad(expected.sum(),(reference_p,source.baseline))
    for a,e in zip(gradients,reference_gradients):torch.testing.assert_close(a,e,rtol=0,atol=0)


def test_hand_computed_outputs_and_gradients():
    p=torch.tensor([[.2,.3]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([0.],dtype=torch.float64,requires_grad=True)
    result=noisy_or_with_baseline(p,b)
    torch.testing.assert_close(result,torch.tensor([.72],dtype=torch.float64))
    gp,gb=torch.autograd.grad(result.sum(),(p,b))
    torch.testing.assert_close(gp,torch.tensor([[.35,.4]],dtype=torch.float64))
    torch.testing.assert_close(gb,torch.tensor([.14],dtype=torch.float64))
    assert torch.autograd.gradcheck(noisy_or_with_baseline,(p,b))


def test_boundary_probabilities_and_input_retention():
    p=torch.tensor([[0.,0.],[0.,1.],[1.,1.]],requires_grad=True)
    before=p.detach().clone();b=torch.tensor([0.],requires_grad=True)
    result=noisy_or_with_baseline(p,b)
    torch.testing.assert_close(result,torch.tensor([.5,1.,1.]),rtol=0,atol=0)
    gp,gb=torch.autograd.grad(result.sum(),(p,b))
    torch.testing.assert_close(gp,torch.tensor([[.5,.5],[0.,.5],[0.,0.]]),rtol=0,atol=0)
    torch.testing.assert_close(gb,torch.tensor([.25]),rtol=0,atol=0)
    torch.testing.assert_close(p,before,rtol=0,atol=0)


@pytest.mark.parametrize('p,b',[
    (torch.ones(2),torch.zeros(1)),(torch.empty(2,0),torch.zeros(1)),
    (torch.tensor([[1.1]]),torch.zeros(1)),(torch.tensor([[float('nan')]]),torch.zeros(1)),
    (torch.ones(1,1,dtype=torch.int64),torch.zeros(1)),(torch.ones(1,1),torch.zeros(2)),
    (torch.ones(1,1),torch.zeros(1,dtype=torch.float64)),(torch.ones(1,1),torch.tensor([float('inf')])),
])
def test_invalid_contracts(p,b):
    with pytest.raises(ValueError):noisy_or_with_baseline(p,b)


def test_witness_shape_contract():
    output=witness_noisy_or_with_baseline(AbstractArray(shape=(4,7),dtype='float32'),AbstractArray(shape=(1,),dtype='float32'))
    assert output.shape==(4,) and output.min_val==0 and output.max_val==1

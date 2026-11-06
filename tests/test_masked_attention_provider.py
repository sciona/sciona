"""Synthetic analytic values, derivatives and model-policy parity."""
import math
import pytest
import torch
from sciona.atoms.ml.masked_attention import masked_attention_pool, sigmoid_context_gate
from sciona.video_multilabel_network import VideoHead
from sciona.video_multilabel_reusable_network import ReusableVideoHead


def test_analytic_pool_and_nonfinite_padding():
    values = torch.tensor([[[2., 4.], [6., 8.], [float('nan'), float('inf')]]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([[0., math.log(3), float('nan')]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    pooled, weights = masked_attention_pool(values, logits, mask)
    torch.testing.assert_close(pooled, torch.tensor([[5., 7.]], dtype=torch.float64))
    torch.testing.assert_close(weights, torch.tensor([[.25, .75, 0.]], dtype=torch.float64))
    pooled.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[[.25,.25],[.75,.75],[0.,0.]]], dtype=torch.float64))
    torch.testing.assert_close(logits.grad, torch.tensor([[-1.5,1.5,0.]], dtype=torch.float64))
    assert torch.isnan(values[0,2,0]) and torch.isnan(logits[0,2])


def test_numerical_derivatives_and_permutation():
    values = torch.arange(12, dtype=torch.float64).reshape(2,3,2).requires_grad_()
    logits = torch.tensor([[.2,-.1,.8],[-.2,.3,.4]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True,False,True],[False,True,True]])
    assert torch.autograd.gradcheck(lambda v,l: masked_attention_pool(v,l,mask), (values,logits))
    pooled, weights = masked_attention_pool(values,logits,mask)
    order = [2,0,1]
    permuted, permuted_weights = masked_attention_pool(values[:,order],logits[:,order],mask[:,order])
    torch.testing.assert_close(permuted,pooled)
    torch.testing.assert_close(permuted_weights,weights[:,order])
    gate_logits = torch.zeros_like(pooled, requires_grad=True)
    assert torch.autograd.gradcheck(sigmoid_context_gate,(pooled,gate_logits))
    torch.testing.assert_close(sigmoid_context_gate(pooled,gate_logits),pooled/2)


@pytest.mark.parametrize('case', ['empty','all_masked','bad_mask','bad_logits','valid_nan','logit_inf','dtype'])
def test_invalid_pool_contract(case):
    values = torch.ones((2,3,2),dtype=torch.float64)
    logits = torch.zeros((2,3),dtype=torch.float64)
    mask = torch.ones((2,3),dtype=torch.bool)
    if case == 'empty': values=values[:0];logits=logits[:0];mask=mask[:0]
    if case == 'all_masked': mask[0]=False
    if case == 'bad_mask': mask=mask.double()
    if case == 'bad_logits': logits=logits[:,:2]
    if case == 'valid_nan': values[0,0,0]=float('nan')
    if case == 'logit_inf': logits[0,0]=float('inf')
    if case == 'dtype': logits=logits.float()
    with pytest.raises(ValueError): masked_attention_pool(values,logits,mask)


@pytest.mark.parametrize('case', ['empty','shape','dtype','nonfinite','integer'])
def test_invalid_gate_contract(case):
    values=torch.ones((2,3),dtype=torch.float64);logits=values.clone()
    if case == 'empty': values=values[:0];logits=logits[:0]
    if case == 'shape': logits=logits[:,:2]
    if case == 'dtype': logits=logits.float()
    if case == 'nonfinite': logits[0,0]=float('inf')
    if case == 'integer': values=values.long();logits=logits.long()
    with pytest.raises(ValueError): sigmoid_context_gate(values,logits)


def test_model_outputs_and_parameter_gradients_match_reference():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        reference=VideoHead(2,3,2).double()
        calls=[]
        def pool(*args):
            calls.append('pool');return masked_attention_pool(*args)
        def gate(*args):
            calls.append('gate');return sigmoid_context_gate(*args)
        actual=ReusableVideoHead(2,3,2,pool_function=pool,gate_function=gate).double()
        actual.load_state_dict(reference.state_dict())
    frames=torch.arange(12,dtype=torch.float64).reshape(2,3,2)
    mask=torch.tensor([[True,False,True],[False,True,True]])
    frames[~mask]=float('nan')
    expected=reference(frames,mask);observed=actual(frames,mask)
    torch.testing.assert_close(observed,expected,rtol=0,atol=0)
    expected.sum().backward();observed.sum().backward()
    for p,q in zip(reference.parameters(),actual.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert calls==['pool','gate']

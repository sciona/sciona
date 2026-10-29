import pytest
import torch
from sciona.atoms.dl.loss.ranked_selection import highest_scoring_indices


def test_explicit_candidates_order_and_gather_gradients():
    scores=torch.tensor([4.,9.,-2.,3.,8.],requires_grad=True)
    candidates=torch.tensor([4,0,3])
    selected=highest_scoring_indices(scores,candidates,2)
    torch.testing.assert_close(selected,torch.tensor([4,0]),rtol=0,atol=0)
    scores[selected].sum().backward()
    torch.testing.assert_close(scores.grad,torch.tensor([1.,0.,0.,0.,1.]),rtol=0,atol=0)
    torch.testing.assert_close(candidates,torch.tensor([4,0,3]),rtol=0,atol=0)


@pytest.mark.parametrize('k',[0,1,3,10])
def test_ties_retain_native_torch_order(k):
    scores=torch.tensor([2.,0.,2.,2.,-1.]);candidates=torch.tensor([3,0,2])
    expected=candidates[torch.topk(scores[candidates],min(k,3)).indices]
    torch.testing.assert_close(highest_scoring_indices(scores,candidates,k),expected,rtol=0,atol=0)


def test_empty_candidates_and_score_vector():
    for scores in [torch.empty(0),torch.ones(3)]:
        actual=highest_scoring_indices(scores,torch.empty(0,dtype=torch.int64),4)
        assert actual.shape==(0,) and actual.dtype==torch.int64


@pytest.mark.parametrize('indices',[[0,0],[-1],[3]])
def test_bad_candidate_values(indices):
    with pytest.raises(ValueError):highest_scoring_indices(torch.ones(3),torch.tensor(indices),1)


@pytest.mark.parametrize('k',[-1,True,1.2])
def test_bad_selection_count(k):
    with pytest.raises(ValueError):highest_scoring_indices(torch.ones(3),torch.tensor([0]),k)


def test_bad_tensors():
    for scores,indices in [(torch.tensor([float('nan')]),torch.tensor([0])),
        (torch.ones(2,2),torch.tensor([0])),(torch.ones(2,dtype=torch.int64),torch.tensor([0])),
        (torch.ones(2),torch.tensor([0.]))]:
        with pytest.raises(ValueError):highest_scoring_indices(scores,indices,1)

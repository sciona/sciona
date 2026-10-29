import pytest
import torch
from sciona.dsb_decomposed_losses import classifier_loss
from sciona.dsb_losses import classifier_loss as reference_loss
from sciona.dsb_decomposed_loss_graph import build_dsb_loss_graph


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('ratio',[0.,.5,1.,2.])
def test_composed_loss_and_all_gradients_match_source_recipe(dtype,ratio):
    case=torch.tensor([.2,.7],dtype=dtype,requires_grad=True)
    probabilities=torch.tensor([[0.,.02,.03],[.2,.5,1.]],dtype=dtype,requires_grad=True)
    weights=torch.tensor([[1.,.5,1.],[0.,1.,1.]],dtype=dtype,requires_grad=True)
    labels=torch.tensor([0.,1.],dtype=dtype)
    reference_inputs=[v.detach().clone().requires_grad_() for v in [case,probabilities,weights]]
    expected=reference_loss(reference_inputs[0],reference_inputs[1],labels,reference_inputs[2],miss_ratio=ratio)
    actual=classifier_loss(case,probabilities,labels,weights,miss_ratio=ratio)
    for key in ['total','classification','miss']:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    for a,b in zip(torch.autograd.grad(actual['total'],(case,probabilities,weights)),
                   torch.autograd.grad(expected['total'],reference_inputs)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_generic_penalty_receives_source_controls(monkeypatch):
    from sciona import dsb_decomposed_losses as module
    original=module.thresholded_log_penalty;calls=[]
    def spy(p,w,threshold,offset):
        calls.append((threshold,offset))
        return original(p,w,threshold,offset)
    monkeypatch.setattr(module,'thresholded_log_penalty',spy)
    classifier_loss(torch.tensor([.2]),torch.tensor([[.01,.02]]),torch.tensor([0.]),torch.ones(1,2))
    assert calls==[(.03,.001)]


def test_graph_routes_training_through_new_loss_recipe():
    from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
    graph=build_dsb_loss_graph();digest,nodes,edges=encode_execution_graph(graph)
    assert len(nodes)==19 and decode_execution_graph(nodes,edges,digest)==graph
    training=next(n for n in graph.nodes if n.node_id=='training')
    assert training.matched_primitive=='sciona.atoms.dl.detection.dsb_loss_training.train_shared_runtime'
    assert {p.name for p in training.outputs}=={'classifier_checkpoint','metrics'}
    assert graph.metadata['publication_status']=='draft'

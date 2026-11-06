"""Model-only checkpoint boundaries preserve live training state."""
import copy
import io
import pytest
import torch
from sciona.atoms.dl.detection.dsb_lifecycle_steps import export_model_checkpoint, concrete_only


def test_checkpoint_is_detached_snapshot_without_optimizer_or_rng():
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    model(torch.ones(2,2)).sum().backward();optimizer.step()
    expected=copy.deepcopy(model.state_dict())
    optimizer_state=copy.deepcopy(optimizer.state_dict())
    rng=torch.get_rng_state().clone()
    payload=export_model_checkpoint(model,7)
    checkpoint=torch.load(io.BytesIO(payload),weights_only=True)
    assert set(checkpoint)=={'epoch','state_dict'} and checkpoint['epoch']==7
    torch.testing.assert_close(checkpoint['state_dict'],expected,rtol=0,atol=0)
    torch.testing.assert_close(model.state_dict(),expected,rtol=0,atol=0)
    torch.testing.assert_close(optimizer.state_dict(),optimizer_state,rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert model.training
    assert all(v.device.type=='cpu' and not v.requires_grad for v in checkpoint['state_dict'].values())
    with torch.no_grad():model.weight.add_(1)
    torch.testing.assert_close(torch.load(io.BytesIO(payload),weights_only=True)['state_dict'],expected,rtol=0,atol=0)


@pytest.mark.parametrize('epoch',[-1,True,1.5])
def test_invalid_checkpoint_epoch(epoch):
    with pytest.raises(ValueError):export_model_checkpoint(torch.nn.Linear(2,1),epoch)


def test_no_abstract_model_claim():
    with pytest.raises(ValueError,match='Concrete model'):concrete_only()

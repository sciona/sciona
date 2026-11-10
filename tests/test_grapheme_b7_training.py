"""Synthetic update-oracle checks; these small models do not qualify full B7."""
import copy
import pytest
import torch
from torch import nn
from sciona.grapheme_b7_training import make_optimizer,train_batch


class SmallClassifier(nn.Module):
    def __init__(self,branch):
        super().__init__();self.branch=branch
        self.fc=nn.Linear(3,1295 if branch=='ood' else 14784)

    def forward(self,x):return self.fc(x.mean((2,3)))


@pytest.mark.parametrize('branch',['seen','ood'])
def test_native_updates_match_explicit_loss_and_optimizer_reference(branch):
    torch.set_num_threads(1);torch.manual_seed(421)
    actual=SmallClassifier(branch);reference=copy.deepcopy(actual)
    optimizer=make_optimizer(actual)
    other=torch.optim.AdamW(reference.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
                            weight_decay=.01,amsgrad=False)
    images=torch.rand(2,3,137,236);labels=torch.tensor([0,1])
    initial=actual.fc.weight.detach().clone()
    for index in (0,1,2,3,4):
        result=train_batch(actual,optimizer,images,labels,branch=branch,schedule_index=index,
                           total_steps=5,warmup_steps=2)
        other.zero_grad(set_to_none=True);logits=reference(images)
        if branch=='seen':loss=nn.functional.cross_entropy(logits,labels)
        else:
            targets=nn.functional.one_hot(labels,1295).float()
            selected=(targets.bool()|(logits.detach().sigmoid()>.1)).float()
            loss=((nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')*selected).sum(1)/selected.sum(1)).mean()
        loss.backward()
        other.param_groups[0]['lr']=.001*(index/2 if index<2 else (5-index)/3)
        other.step()
        assert result['loss']==float(loss.detach())
        assert result['learning_rate']==other.param_groups[0]['lr']
        for a,b in zip(actual.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(initial,actual.fc.weight)


@pytest.mark.parametrize('fault',['label','shape','nonfinite','branch','index','warmup','optimizer_owner','optimizer_settings'])
def test_invalid_inputs_rejected_before_model_or_optimizer_mutation(fault):
    model=SmallClassifier('ood').eval();optimizer=make_optimizer(model)
    images=torch.ones(2,3,137,236);labels=torch.tensor([0,1])
    options=dict(branch='ood',schedule_index=1,total_steps=5,warmup_steps=2)
    if fault=='label':labels[1]=1295
    elif fault=='shape':images=images[:,:,:32,:32]
    elif fault=='nonfinite':images[0,0,0,0]=float('nan')
    elif fault=='branch':options['branch']='seen'
    elif fault=='index':options['schedule_index']=True
    elif fault=='warmup':options['warmup_steps']=5
    elif fault=='optimizer_owner':optimizer=make_optimizer(SmallClassifier('ood'))
    else:optimizer.param_groups[0]['weight_decay']=0
    before={k:v.clone() for k,v in model.state_dict().items()}
    rate=optimizer.param_groups[0]['lr']
    with pytest.raises(ValueError):train_batch(model,optimizer,images,labels,**options)
    assert not model.training and not optimizer.state and optimizer.param_groups[0]['lr']==rate
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])

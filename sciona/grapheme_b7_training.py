"""Connect recovered classifier losses and AdamW to an explicit schedule index.

Inputs are already normalized by the caller. The caller supplies the index of
the learning-rate multiplier used for this update; historical scheduler order
and warmup duration are not inferred. This is not full winner-workflow evidence.
Failures after forward execution may change model buffers; abort such a run.
"""
import torch
from torch import nn
from sciona.bengali_ood_loss import ood_training_loss
from sciona.grapheme_b7_schedule import warmup_linear_multiplier


def _parameters(model):
    if not isinstance(model,nn.Module):
        raise ValueError('Trainable classifier module required')
    parameters=list(model.parameters())
    if not parameters or any(p.device.type!='cpu' or p.dtype!=torch.float32 or not p.requires_grad
                             for p in parameters):
        raise ValueError('Trainable float32 CPU classifier parameters required')
    return parameters


def make_optimizer(model):
    """Use the AdamW settings reported for both winner classifier branches."""
    return torch.optim.AdamW(_parameters(model),lr=.001,betas=(.9,.999),eps=1e-8,
                             weight_decay=.01,amsgrad=False)


def train_batch(model,optimizer,images,labels,*,branch,schedule_index,total_steps,warmup_steps):
    """Perform one native update; supplied models are not authenticated as B7."""
    parameters=_parameters(model)
    if branch not in ('ood','seen') or getattr(model,'branch',branch)!=branch:
        raise ValueError('Consistent seen or ood classifier branch required')
    classes=1295 if branch=='ood' else 14784
    if (not isinstance(images,torch.Tensor) or images.device.type!='cpu' or images.dtype!=torch.float32
            or images.ndim!=4 or images.shape[0]<1 or images.shape[1:]!=(3,137,236)
            or not torch.isfinite(images).all()):
        raise ValueError('Finite normalized float32 CPU N3x137x236 input required')
    if (not isinstance(labels,torch.Tensor) or labels.device.type!='cpu' or labels.dtype!=torch.int64
            or labels.shape!=(len(images),) or (labels<0).any() or (labels>=classes).any()):
        raise ValueError('Aligned in-range integer classifier targets required')
    multiplier=warmup_linear_multiplier(schedule_index,total_steps=total_steps,warmup_steps=warmup_steps)
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups)!=1:
        raise ValueError('Single-group source AdamW optimizer required')
    group=optimizer.param_groups[0]
    if ([id(p) for p in group['params']]!=[id(p) for p in parameters]
            or group['betas']!=(.9,.999) or group['eps']!=1e-8
            or group['weight_decay']!=.01 or group['amsgrad'] or group.get('maximize',False)):
        raise ValueError('Optimizer ownership or source settings differ')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits=model(images)
    if (logits.shape!=(len(images),classes) or logits.device.type!='cpu'
            or logits.dtype!=torch.float32 or not torch.isfinite(logits).all()):
        raise ValueError('Finite branch-sized classifier logits required')
    loss=ood_training_loss(logits,labels) if branch=='ood' else nn.functional.cross_entropy(logits,labels)
    if not torch.isfinite(loss):raise ValueError('Finite classifier loss required')
    loss.backward()
    gradients=[p.grad for p in parameters if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError('Finite classifier gradients required')
    group['lr']=.001*multiplier
    optimizer.step()
    return dict(loss=float(loss.detach()),learning_rate=group['lr'],schedule_index=schedule_index,
                examples=len(images),branch=branch)

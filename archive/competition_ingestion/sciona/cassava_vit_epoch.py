"""CPU ViT epoch mechanics reconstructed from pinned training source.

This does not provide TPU replication, model initialization, augmentation,
fold construction or checkpoint selection. Losses are summed across two
microbatches, including a final short accumulation group, as in the source.
"""
import torch

from sciona.cassava_loss import vit_loss


def train_epoch(model, batches, optimizer, scheduler=None, *, loss_function=vit_loss):
    """Train one finite CPU epoch and advance the scheduler once at its end.

Inputs are already transformed float32 batches and integer class indices.
Callers own initial gradient state, matching the source's epoch boundary.
"""
    if not isinstance(batches, (tuple, list)) or not batches:
        raise ValueError('A nonempty materialized batch sequence is required')
    for inputs, labels in batches:
        if (not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor)
                or inputs.device.type != 'cpu' or inputs.dtype != torch.float32
                or inputs.ndim < 2 or inputs.shape[0] < 1
                or labels.device.type != 'cpu' or labels.dtype != torch.int64
                or labels.shape != (inputs.shape[0],)
                or not torch.isfinite(inputs).all() or (labels < 0).any() or (labels >= 5).any()):
            raise ValueError('Finite CPU float32 batches and integer five-class labels required')
    model.train()
    steps = 0
    for index, (inputs, labels) in enumerate(batches):
        logits = model(inputs)
        targets = torch.nn.functional.one_hot(labels, num_classes=5).to(logits.dtype)
        loss_function(logits, targets).mean().backward()
        if (index + 1) % 2 == 0 or index + 1 == len(batches):
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
    if scheduler is not None:
        scheduler.step()
    return {'batches': len(batches), 'optimizer_steps': steps}

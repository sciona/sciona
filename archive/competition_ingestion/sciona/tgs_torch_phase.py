"""Integrated single-fold PyTorch TGS phase; full source orchestration pending."""
import copy
import numpy as np
import torch

from sciona.tgs_augmentation import augment_torch
from sciona.tgs_preprocessing import prepare_torch
from sciona.tgs_losses import pytorch_training_loss
from sciona.hubmap_losses import binary_lovasz_hinge
from sciona.tgs_metrics import torch_validation_metric
from sciona.tgs_training_state import optimizer_for


def train_phase(model, training_images, training_masks, validation_images, validation_masks,
                *, controls, rng):
    """Train with shuffled passes, cropped validation and cycle-best snapshots.

    Caller initializes each fold explicitly. Each cosine cycle resets SGD and
    its momentum. Cycle-best tensors are cloned; the first finite metric is
    eligible even when zero, correcting the source's uninitialized/stale best.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError('explicit NumPy Generator required')
    epochs, batch, snapshots = [int(controls[k]) for k in ('epoch', 'batch_size', 'snapshot')]
    checkpoint_policy = controls.get('checkpoint_policy', 'cycle_best')
    if checkpoint_policy not in ('cycle_best', 'cycle_end'):
        raise ValueError('unknown source checkpoint policy')
    if epochs < 1 or batch < 1 or not 1 <= snapshots <= epochs or epochs % snapshots:
        raise ValueError('positive schedule with complete equal cycles required')
    maximum, minimum = float(controls['max_lr']), float(controls['min_lr'])
    if not np.isfinite(minimum) or not 0 <= minimum <= maximum:
        raise ValueError('invalid minimum learning rate')
    train, target, valid, valid_target = map(np.asarray, (training_images, training_masks, validation_images, validation_masks))
    for images, masks in ((train, target), (valid, valid_target)):
        if not len(images):
            raise ValueError('nonempty training/validation populations required')
        for i in range(0, len(images), batch):
            prepare_torch(images[i:i+batch], masks[i:i+batch], variant=model.variant)
    optimizer = optimizer_for(model, 'pytorch', maximum)
    period = epochs // snapshots
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, period, eta_min=minimum)
    history, states = [], {}
    best_score, best_state = -float('inf'), None
    updates = 0
    def weights():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        rate = optimizer.param_groups[0]['lr']
        order = rng.permutation(len(train))
        model.train()
        total_loss = 0.
        for start in range(0, len(train), batch):
            selected = order[start:start+batch]
            augmented = [augment_torch(train[i], target[i], rng) for i in selected]
            prepared = prepare_torch(np.stack([v[0] for v in augmented]), np.stack([v[1] for v in augmented]), variant=model.variant, training=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(prepared['images'])
            if model.variant == 3:
                loss = pytorch_training_loss(output[0], prepared['masks'], pixel_logits=output[1],
                                              image_probabilities=output[2], image_targets=prepared['empty_image_targets'])
            else:
                loss = pytorch_training_loss(output, prepared['masks'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('nonfinite training gradient')
            optimizer.step()
            updates += 1
            total_loss += float(loss.detach()) * len(selected)
        model.eval()
        validation_loss, metric = 0., 0.
        with torch.no_grad():
            for start in range(0, len(valid), batch):
                prepared = prepare_torch(valid[start:start+batch], valid_target[start:start+batch], variant=model.variant)
                output = model(prepared['images'])
                if model.variant == 3:
                    output = output[0]
                a, b = prepared['crop_start'], prepared['crop_start'] + prepared['crop_size']
                output = output[:, :, a:b, a:b].contiguous()
                validation_loss += float(binary_lovasz_hinge(output, prepared['masks'])) * len(output)
                metric += torch_validation_metric(output, prepared['masks'])['mean'] * len(output)
        metric /= len(valid)
        validation_loss /= len(valid)
        if not np.isfinite(validation_loss):
            raise ValueError('nonfinite validation loss')
        scheduler.step()
        if metric > best_score:
            best_score, best_state = metric, weights()
        if (epoch + 1) % period == 0:
            states[epoch + 1] = weights() if checkpoint_policy == 'cycle_end' else best_state
            best_score, best_state = -float('inf'), None
            optimizer = optimizer_for(model, 'pytorch', maximum)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, period, eta_min=minimum)
        history.append(dict(epoch=epoch, learning_rate=rate, training_loss=total_loss/len(train),
                            validation_loss=validation_loss, metric=metric))
    return dict(cycle_states=states, history=history, actual_epochs=len(history), optimizer_updates=updates,
                rng_state=copy.deepcopy(rng.bit_generator.state))

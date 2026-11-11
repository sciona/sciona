"""Integrated single-phase TGS Keras-branch training in the qualified CPU model.

Full workflow must supply reviewed initialization, fold selection, source phase
controls, and checkpoint handoffs. This module does not declare CDG completion.
"""
import copy
import numpy as np
import torch

from sciona.tgs_keras_augmentation import augment_keras
from sciona.tgs_keras_preprocessing import prepare_keras
from sciona.tgs_losses import keras_bce_dice, keras_elu_lovasz
from sciona.tgs_metrics import keras_validation_metric
from sciona.tgs_training_state import optimizer_for, snapshot_learning_rate
from sciona.tgs_callbacks import ValidationControl


def train_phase(model, training_images, training_masks, validation_images, validation_masks,
                *, controls, rng):
    """Run source batch sampling, augmentation, losses, validation and callbacks.

    Model initialization and the NumPy generator are explicit. Counts below one
    complete training batch fail instead of producing a zero-step phase.
    Best and cosine snapshots contain isolated model tensors only, consistent
    with source phase handoffs that construct a new optimizer.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError('explicit augmentation/sampling Generator required')
    epochs, batch = int(controls['epochs']), int(controls['batch_size'])
    if epochs < 1 or batch < 1:
        raise ValueError('positive epoch and batch counts required')
    train, target = np.asarray(training_images), np.asarray(training_masks)
    valid, valid_target = np.asarray(validation_images), np.asarray(validation_masks)
    # Validate full populations before the first model update.
    for images, masks in ((train, target), (valid, valid_target)):
        if not len(images):
            raise ValueError('nonempty training and validation populations required')
        for i in range(0, len(images), batch):
            prepare_keras(images[i:i+batch], masks[i:i+batch])
    steps = 2 * (len(train) // batch)
    if not steps:
        raise ValueError('source schedule requires at least one complete batch')
    loss_name = controls['loss_function']
    if loss_name not in ('lovasz', 'bce_dice'):
        raise ValueError('unknown source loss')
    model.probabilities = loss_name == 'bce_dice'
    loss_fn = keras_bce_dice if model.probabilities else keras_elu_lovasz
    rate = float(controls['learning_rate'])
    optimizer = optimizer_for(model, 'keras', rate)
    callback = controls['callback']
    monitor = None
    if callback == 'reduce_lr':
        monitor = ValidationControl(learning_rate=rate, stop_patience=int(controls['early_stop_patience']),
                                    reduce_patience=int(controls['reduce_lr_patience']), factor=float(controls['reduce_lr_factor']),
                                    minimum=float(controls['reduce_lr_min']))
    elif callback != 'snapshot':
        raise ValueError('unknown phase callback')
    snapshots = int(controls.get('n_snapshots', 1))
    if callback == 'snapshot':
        snapshot_learning_rate(0, epochs, snapshots, rate)
    best_metric, best_state = -float('inf'), None
    periodic, history = {}, []
    def weights():
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(epochs):
        if callback == 'snapshot':
            optimizer.param_groups[0]['lr'] = snapshot_learning_rate(epoch, epochs, snapshots, rate)
        epoch_rate = optimizer.param_groups[0]['lr']
        model.train()
        training_loss = 0.
        for _ in range(steps):
            selected = rng.integers(0, len(train), size=batch)
            augmented = [augment_keras(train[i], target[i], rng) for i in selected]
            prepared = prepare_keras(np.stack([pair[0] for pair in augmented]), np.stack([pair[1] for pair in augmented]))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(prepared['images']), prepared['masks'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError('nonfinite training gradient')
            optimizer.step()
            training_loss += float(loss.detach())
        model.eval()
        totals = np.zeros(3)
        with torch.no_grad():
            for start in range(0, len(valid), batch):
                prepared = prepare_keras(valid[start:start+batch], valid_target[start:start+batch])
                prediction = model(prepared['images'])
                metric_args = dict(score_kind='probabilities' if model.probabilities else 'logits')
                capped = keras_validation_metric(prediction, prepared['masks'], source_union_cap=True, **metric_args)['mean']
                uncapped = keras_validation_metric(prediction, prepared['masks'], source_union_cap=False, **metric_args)['mean']
                totals += len(prediction) * np.array([float(loss_fn(prediction, prepared['masks'])), capped, uncapped])
        totals /= len(valid)
        if not np.isfinite(totals).all():
            raise ValueError('nonfinite validation result')
        if totals[1] > best_metric:
            best_metric, best_state = float(totals[1]), weights()
        decision = monitor.observe(float(totals[1])) if monitor else dict(stop=False, learning_rate=epoch_rate)
        optimizer.param_groups[0]['lr'] = decision['learning_rate']
        if callback == 'snapshot' and epoch != 0 and (epoch + 1) % (epochs // snapshots) == 0:
            periodic[epoch + 1] = weights()
        history.append(dict(epoch=epoch, learning_rate=epoch_rate, training_loss=training_loss / steps,
                            validation_loss=float(totals[0]), source_metric=float(totals[1]), uncapped_metric=float(totals[2])))
        if decision['stop']:
            break
    return dict(best_state=best_state, periodic_states=periodic, history=history,
                actual_epochs=len(history), optimizer_updates=len(history) * steps,
                steps_per_epoch=steps, rng_state=copy.deepcopy(rng.bit_generator.state))

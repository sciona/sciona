"""Declared classifier lifecycle; caller owns preprocessing and validation.

No historical warmup, normalization, split or checkpoint rule is inferred.
Checkpoints contain inference weights, not resumable optimizer/RNG state.
Failures can leave partial model updates and files: discard failed runs.
"""
import hashlib
import math
from pathlib import Path

import torch

from sciona.grapheme_b7_schedule import warmup_linear_multiplier
from sciona.grapheme_b7_training import make_optimizer, train_batch


def fit_classifier(model, *, branch, epochs, steps_per_epoch, batches_for_epoch,
                   validate, warmup_steps, first_schedule_index, checkpoint_policy,
                   output_directory):
    """Consume exactly the declared batches and save caller-selected weights.

    batches_for_epoch(epoch) yields (normalized images, integer targets).
    validate(model, epoch) supplies a finite scalar score to maximize. Its
    split, metric and preprocessing must be qualified separately. It runs in
    eval mode with gradients disabled, once after each training epoch.
    """
    if (type(epochs) is not int or epochs < 1 or type(steps_per_epoch) is not int
            or steps_per_epoch < 1 or type(first_schedule_index) is not int
            or first_schedule_index not in (0, 1)):
        raise ValueError('Positive epoch/step counts and explicit schedule origin 0 or 1 required')
    if (branch not in ('seen', 'ood') or getattr(model, 'branch', branch) != branch
            or not callable(batches_for_epoch) or not callable(validate)
            or checkpoint_policy not in ('last', 'max_validation')):
        raise ValueError('Consistent branch, batch/validation callbacks and explicit checkpoint policy required')
    total_steps = epochs * steps_per_epoch
    warmup_linear_multiplier(first_schedule_index, total_steps=total_steps, warmup_steps=warmup_steps)
    optimizer = make_optimizer(model)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    history = []
    best_score = -math.inf
    selected = None
    completed_steps = 0
    for epoch in range(epochs):
        batches = iter(batches_for_epoch(epoch))
        total_loss = 0.
        examples = 0
        for _ in range(steps_per_epoch):
            try:
                images, labels = next(batches)
            except StopIteration as error:
                raise ValueError('Training epoch ended before declared batch count') from error
            update = train_batch(model, optimizer, images, labels, branch=branch,
                                 schedule_index=first_schedule_index + completed_steps,
                                 total_steps=total_steps, warmup_steps=warmup_steps)
            total_loss += update['loss'] * update['examples']
            examples += update['examples']
            completed_steps += 1
        sentinel = object()
        if next(batches, sentinel) is not sentinel:
            raise ValueError('Training epoch exceeded declared batch count')
        model.eval()
        with torch.no_grad():
            score = validate(model, epoch)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Finite scalar validation score required')
        history.append(dict(epoch=epoch, examples=examples, steps=steps_per_epoch,
                            training_loss=total_loss/examples, validation_score=score,
                            last_learning_rate=update['learning_rate']))
        if ((checkpoint_policy == 'last' and epoch == epochs - 1)
                or (checkpoint_policy == 'max_validation' and score > best_score)):
            partial = directory/'selected.pt.partial'
            torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, partial)
            partial.replace(directory/'selected.pt')
            selected = dict(epoch=epoch, validation_score=score,
                            sha256=hashlib.sha256((directory/'selected.pt').read_bytes()).hexdigest())
            best_score = score
    return dict(completed_epochs=len(history), completed_steps=completed_steps,
                checkpoint_policy=checkpoint_policy, selected_checkpoint=selected,
                history=history, approved=False, first_schedule_index=first_schedule_index,
                warmup_steps=warmup_steps, resumable_checkpoint=False)

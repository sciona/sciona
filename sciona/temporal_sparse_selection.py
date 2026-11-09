"""Repeated causal stream passes with validation-selected checkpoints.

Factories must replay immutable, ordered training and validation populations.
Only training labels update estimators. Each epoch rebuilds its causal encoder
from empty history; no prior-epoch target history leaks into early rows. The
validation score is used for selection and is not an independent accuracy test.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from sciona.temporal_sparse_features import TemporalSparseEncoder, validate_event
from sciona.temporal_sparse_training import fit, clipped
from sciona.atoms.ml.validation_stopping import initialize, observe


@dataclass
class SelectedEnsemble:
    model: object
    validation_scores: tuple[float, ...]
    selected_epoch: int
    stopped_early: bool


def fit_selected(training_factory, validation_factory, *, encoder_controls,
                 controls, max_epochs, patience, min_delta):
    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError('Positive integer epoch budget required')
    policy = initialize(patience, min_delta, 'min')
    current = fit(training_factory(), encoder_controls=encoder_controls, controls=controls)
    best = None
    scores = []
    for epoch in range(max_epochs):
        if epoch:
            encoder = TemporalSparseEncoder(**encoder_controls)

            def checked():
                for event in training_factory():
                    validate_event(event, labeled=True)
                    if event['target'] < 0:
                        raise ValueError('Nonnegative targets required')
                    yield event

            rows = updates = 0
            for matrix, target in encoder.encode(checked(), chunk_size=controls['chunk_size'], labeled=True):
                x = clipped(matrix, controls['feature_clip'])
                for model in current.models:
                    model.partial_fit(x, np.log1p(target))
                    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
                        raise ValueError('Nonfinite online model')
                rows += len(target)
                updates += 1
            if rows != current.training_rows:
                raise ValueError('Replayed training population changed')
            current.encoder = encoder
            current.updates_per_model += updates
        score = current.evaluate(validation_factory())['rmsle']
        scores.append(score)
        policy = observe(policy, score)
        if policy.best_index == epoch:
            best = deepcopy(current)
        if policy.stopped:
            break
    return SelectedEnsemble(best, tuple(scores), policy.best_index + 1, policy.stopped)

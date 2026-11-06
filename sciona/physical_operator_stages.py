"""Separate lifecycle stages for the explicit periodic diffusion recipe.

Physical units, populations, model selection and conservation policy belong to
this recipe. Projection and smoothing delegate to domain-independent numeric
operations. Fitted intermediates are trusted in-process objects, not a portable
model format or a boundary for arbitrary caller-constructed objects.
"""
from dataclasses import dataclass
import json
import numpy as np
import torch

from sciona.physical_operator_state import prepare as validate, tensors, finite_json
from sciona.physical_operator_training import fit as train
from sciona.constrained_array_ops import project_nonnegative_fixed_sum, smooth_periodic_rows


@dataclass(frozen=True)
class PreparedOperator:
    configuration: str


@dataclass(frozen=True)
class FittedOperatorStage:
    prepared: PreparedOperator
    fitted: object


@dataclass(frozen=True)
class PredictedOperatorStage:
    fitted_state: FittedOperatorStage
    values: tuple


@dataclass(frozen=True)
class ProjectedOperatorStage:
    fitted_state: FittedOperatorStage
    values: tuple


def prepare(payload):
    p = validate(payload)
    return PreparedOperator(json.dumps(p, allow_nan=False, sort_keys=True, separators=(',', ':')))


def _configuration(prepared):
    if not isinstance(prepared, PreparedOperator):
        raise ValueError('Prepared operator configuration required')
    return validate(json.loads(prepared.configuration))


def fit(prepared):
    """Fit spectral/local blocks and restore the earliest best validation model."""
    p = _configuration(prepared)
    return FittedOperatorStage(prepared, train(p))


def predict(fitted):
    """Predict the configured query rows without fitting or updating the model."""
    if not isinstance(fitted, FittedOperatorStage):
        raise ValueError('Fitted operator stage required')
    p = _configuration(fitted.prepared)
    x, time = tensors(p['query'], p['length'])
    with torch.no_grad():
        values = fitted.fitted.model(x, time).numpy()
    if values.shape != x.shape or not np.isfinite(values).all():
        raise ValueError('Finite aligned query predictions required')
    return PredictedOperatorStage(fitted, tuple(map(tuple, values.tolist())))


def project(predicted):
    """Apply explicit nonnegative initial-sum conservation on the uniform grid."""
    if not isinstance(predicted, PredictedOperatorStage):
        raise ValueError('Predicted operator stage required')
    p = _configuration(predicted.fitted_state.prepared)
    initial = np.asarray(p['query']['states'], dtype=np.float64)
    with np.errstate(over='ignore'):
        totals = initial.sum(axis=1)
    values = project_nonnegative_fixed_sum(predicted.values, totals)
    return ProjectedOperatorStage(predicted.fitted_state, tuple(map(tuple, values.tolist())))


def finish(projected):
    """Apply the explicit periodic smoothing policy and format finite results."""
    if not isinstance(projected, ProjectedOperatorStage):
        raise ValueError('Projected operator stage required')
    state = projected.fitted_state
    p = _configuration(state.prepared)
    fitted = state.fitted
    predictions = smooth_periodic_rows(projected.values, p['controls']['smoothing'])
    initial = np.asarray(p['query']['states'], dtype=np.float64)
    result = dict(predictions=predictions.tolist(), training_rows=len(p['training']['states']),
        validation_rows=len(p['validation']['states']), query_rows=len(initial), grid_points=initial.shape[1],
        epochs=p['controls']['epochs'], best_epoch=fitted.best_epoch, validation_mse=fitted.validation_mse,
        initial_training_mse=fitted.history[0]['training_mse'], final_training_mse=fitted.history[-1]['training_mse'],
        max_mean_conservation_error=float(np.max(np.abs(predictions.mean(1) - initial.mean(1)))))
    finite_json(result)
    return result

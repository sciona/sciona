"""Complete Santa recipe stages with an external ordered numeric combination.

Intermediate values must belong to the same invocation. Structural checks do
not authenticate model training or numeric computation provenance.
"""
import hashlib
import json
import math
import numpy as np

from sciona.santa_execution import Prepared, prepare, _history
from sciona.santa_boosting import ThresholdModels, fit_threshold_models
from sciona.santa_agent_features import exploration_ratio


def checked_payload(prepared):
    if type(prepared) is not Prepared or type(prepared.configuration) is not str:
        raise ValueError('Prepared execution required')
    if hashlib.sha256(prepared.configuration.encode()).hexdigest() != prepared.fingerprint:
        raise ValueError('Prepared execution changed')
    try:
        payload = json.loads(prepared.configuration)
    except (ValueError, TypeError):
        raise ValueError('Invalid prepared JSON') from None
    if prepare(payload) != prepared:
        raise ValueError('Prepared execution is not canonical')
    return payload


def fit(prepared):
    payload = checked_payload(prepared)
    controls = dict(payload['controls'])
    controls.pop('decision_seed')
    return fit_threshold_models(payload['training'], payload['validation'], **controls)


def predictions(prepared, models):
    payload = checked_payload(prepared)
    if type(models) is not ThresholdModels:
        raise ValueError('Complete two-model fit required')
    normal, transformed = _history(payload['query']).features()
    return dict(normal=models.normal.predict(normal,
                    num_iteration=models.normal.best_iteration, num_threads=1),
                transformed=models.transformed.predict(transformed,
                    num_iteration=models.transformed.best_iteration, num_threads=1))


def blend_inputs(prepared, predicted):
    payload = checked_payload(prepared)
    if type(predicted) is not dict or set(predicted) != {'normal', 'transformed'}:
        raise ValueError('Both prediction families required')
    normal = np.asarray(predicted['normal'], dtype=np.float64)
    transformed = np.asarray(predicted['transformed'], dtype=np.float64)
    if (normal.shape != (100,) or transformed.shape != (100,)
            or not np.isfinite(normal).all() or not np.isfinite(transformed).all()
            or (transformed <= 0).any()):
        raise ValueError('Finite predictions and positive transformed predictions required')
    ratio = exploration_ratio(_history(payload['query']).step)
    # Preserve source arithmetic order: transformed term first, normal second.
    return np.stack((np.log(transformed) / math.log(1.02), normal)), np.array([ratio, 1. - ratio])


def result(prepared, models, initial):
    payload = checked_payload(prepared)
    if type(models) is not ThresholdModels:
        raise ValueError('Complete two-model fit required')
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (100,) or not np.isfinite(initial).all():
        raise ValueError('One finite blended value per action required')
    history = _history(payload['query'])
    scores = initial * np.power(.97, history.own + history.opponent)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite blended scores')
    order = np.random.default_rng(payload['controls']['decision_seed']).permutation(100)
    action = int(order[np.argmax(scores[order])]) if scores.max() > 0 else 0
    output = dict(action=action, scores=scores.tolist(), exploration_ratio=exploration_ratio(history.step),
                  query_step=history.step, models=2, training_rows=models.rows['training'],
                  validation_rows=models.rows['validation'],
                  model_evaluation={name: {k: v for k, v in e.items() if k != 'iteration_rmse'}
                                    for name, e in models.evaluation.items()})
    json.dumps(output, allow_nan=False)
    return output

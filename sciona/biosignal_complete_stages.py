"""Explicit subject-separated fitting, decision calibration and query prediction.

The generic empirical-F1 providers own numeric threshold fitting/application.
Signal units, windowing, model, aggregation and population policy stay here.
"""
from dataclasses import dataclass
import json
import numpy as np
from sciona.biosignal_sequence_contract import prepare as validate_payload, finite_json
from sciona.biosignal_sequence_training import fit as train
from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds


@dataclass(frozen=True, repr=False)
class Prepared:
    configuration: str


@dataclass(frozen=True, repr=False)
class Fitted:
    prepared: Prepared
    model: object


@dataclass(frozen=True, repr=False)
class Calibrated:
    fitted: Fitted
    threshold: float
    calibration_rows: int


def prepare(payload):
    checked = validate_payload(payload).payload
    return Prepared(json.dumps(checked, allow_nan=False, sort_keys=True))


def configuration(prepared):
    if type(prepared) is not Prepared:
        raise ValueError('Prepared signal configuration required')
    return validate_payload(json.loads(prepared.configuration)).payload


def fit(prepared):
    p = configuration(prepared)
    model = train(p['training']['signals'], p['training']['labels'], p['controls'])
    return Fitted(prepared, model)


def calibrate(fitted):
    if type(fitted) is not Fitted:
        raise ValueError('Fitted signal state required')
    p = configuration(fitted.prepared)
    scores, _, _ = fitted.model.predict(p['calibration']['signals'])
    thresholds = fit_empirical_f1_thresholds(np.asarray(scores, dtype=np.float64)[:, None],
        np.asarray(p['calibration']['labels'], dtype=np.int64)[:, None])
    return Calibrated(fitted, float(thresholds[0]), len(scores))


def predict(calibrated):
    if type(calibrated) is not Calibrated:
        raise ValueError('Calibrated signal state required')
    fitted = calibrated.fitted
    p = configuration(fitted.prepared)
    scores, counts, tails = fitted.model.predict(p['query']['signals'])
    classes = apply_column_thresholds(np.asarray(scores, dtype=np.float64)[:, None],
        np.asarray([calibrated.threshold], dtype=np.float64))[:, 0].tolist()
    result = dict(scores=scores, classes=classes, threshold=calibrated.threshold,
        training_rows=len(p['training']['signals']), calibration_rows=calibrated.calibration_rows,
        query_rows=len(scores), training_windows=sum(fitted.model.counts), query_windows=sum(counts),
        omitted_query_samples=sum(tails), epochs=p['controls']['epochs'],
        initial_training_loss=fitted.model.history[0], final_training_loss=fitted.model.history[-1])
    finite_json(result)
    return result

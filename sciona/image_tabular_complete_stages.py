"""Explicit image/tabular lifecycle using reusable numeric F1 providers."""
from dataclasses import dataclass
import json
import numpy as np
from sciona.image_tabular_contract import prepare as validate, finite_json
from sciona.image_tabular_training import fit as train
from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds


@dataclass(frozen=True,repr=False)
class Prepared:
    configuration: str


@dataclass(frozen=True,repr=False)
class Fitted:
    prepared: Prepared
    ensemble: object


@dataclass(frozen=True,repr=False)
class Calibrated:
    fitted: Fitted
    threshold: float
    calibration_rows: int


def prepare(payload):
    return Prepared(json.dumps(validate(payload).payload,sort_keys=True,allow_nan=False))


def configuration(prepared):
    if type(prepared) is not Prepared:raise ValueError('Prepared image/tabular configuration required')
    return validate(json.loads(prepared.configuration)).payload


def fit(prepared):
    p=configuration(prepared)
    return Fitted(prepared,train(**p['training'],controls=p['controls']))


def scores(ensemble,part):
    return np.asarray(ensemble.predict(part['images'],part['numeric'],part['categorical']),dtype=np.float64)


def calibrate(fitted):
    if type(fitted) is not Fitted:raise ValueError('Fitted image/tabular state required')
    p=configuration(fitted.prepared)
    values=scores(fitted.ensemble,p['calibration'])
    threshold=fit_empirical_f1_thresholds(values[:,None],np.asarray(p['calibration']['labels'],dtype=np.int64)[:,None])
    return Calibrated(fitted,float(threshold[0]),len(values))


def predict(calibrated):
    if type(calibrated) is not Calibrated:raise ValueError('Calibrated image/tabular state required')
    fitted=calibrated.fitted;p=configuration(fitted.prepared)
    values=scores(fitted.ensemble,p['query'])
    classes=apply_column_thresholds(values[:,None],np.asarray([calibrated.threshold],dtype=np.float64))[:,0]
    result=dict(scores=values.tolist(),classes=classes.tolist(),threshold=calibrated.threshold,
        training_rows=len(p['training']['images']),calibration_rows=calibrated.calibration_rows,query_rows=len(values),
        folds=len(fitted.ensemble.models),oof_rows=len(fitted.ensemble.oof),visual_features=31,epochs=p['controls']['epochs'],
        initial_training_loss=float(np.mean([m.history[0] for m in fitted.ensemble.models])),
        final_training_loss=float(np.mean([m.history[-1] for m in fitted.ensemble.models])))
    finite_json(result)
    return result

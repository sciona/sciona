"""Concrete execution-local stages for the expanded binary tabular recipe."""
from dataclasses import dataclass
import json

from sciona.tabular_complete_contract import PreparedComplete, prepare, _json
from sciona.tabular_complete_training import fit_complete_stacker
from sciona.tabular_ensemble_calibration import calibrate_stack, predict_calibrated


@dataclass(frozen=True)
class FittedComplete:
    configuration: str
    fitted: object


@dataclass(frozen=True)
class CalibratedComplete:
    fitted_state: FittedComplete
    calibrated: object


def fit_stack(prepared):
    if not isinstance(prepared, PreparedComplete): raise ValueError('Prepared complete recipe required')
    validated = prepare(json.loads(prepared.configuration))
    p = json.loads(validated.configuration)
    controls = dict(p['controls']); controls['feature_controls'] = controls.pop('features')
    fitted = fit_complete_stacker(**p['training'], **controls)
    return FittedComplete(validated.configuration, fitted)


def calibrate(fitted):
    if not isinstance(fitted, FittedComplete): raise ValueError('Fitted complete recipe required')
    p = json.loads(fitted.configuration)
    calibrated = calibrate_stack(fitted.fitted, **p['calibration'])
    return CalibratedComplete(fitted, calibrated)


def predict(calibrated):
    if not isinstance(calibrated, CalibratedComplete): raise ValueError('Calibrated complete recipe required')
    p = json.loads(calibrated.fitted_state.configuration)
    fitted = calibrated.fitted_state.fitted
    result = predict_calibrated(calibrated.calibrated, **p['query'])
    result.update(training_rows=len(p['training']['numeric']), calibration_rows=len(p['calibration']['numeric']),
        query_rows=len(p['query']['numeric']), folds=len(fitted.fold_models), models=2,
        forest_trees=len(fitted.base.trees.estimators_), oof_rows=len(fitted.oof_predictions))
    _json(result)
    return result

"""Full M5 lifecycle with independently bound generic numeric reduction."""
import hashlib
import json
import numpy as np
from sciona.m5_execution import Prepared, prepare, unpack, preprocess
from sciona.m5_pooled_training import train
from sciona.m5_forecast import predict_families


def checked_payload(prepared):
    if (type(prepared) is not Prepared or type(prepared.configuration) is not str
            or hashlib.sha256(prepared.configuration.encode()).hexdigest() != prepared.fingerprint):
        raise ValueError('Unchanged prepared input required')
    p = json.loads(prepared.configuration)
    if prepare(p) != prepared:
        raise ValueError('Canonical prepared input required')
    return p


def features(prepared):
    return preprocess(**unpack(checked_payload(prepared)))


def fit(prepared, transformed):
    p = checked_payload(prepared)
    c = p['configuration']
    cutoff = c['first_day'] + len(c['history'][0]) - 1
    models = train(transformed, cutoff, **p['controls'])
    return dict(models=models, cutoff=cutoff)


def forecast(transformed, fitted):
    return predict_families(transformed, fitted['models'], fitted['cutoff'])


def blend_inputs(forecasted):
    if type(forecasted) is not dict:
        raise ValueError('Complete family forecasts required')
    values = np.asarray(forecasted['families'])
    if (values.ndim != 3 or values.shape[0] != 6 or values.shape[1] < 1 or values.shape[2] != 28
            or values.dtype != np.float64 or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Six aligned finite nonnegative float64 family forecasts required')
    return np.array(values.reshape(6, -1), order='C', copy=True), np.ones(6)


def result(prepared, forecasted, total):
    p = checked_payload(prepared)
    blend_inputs(forecasted)
    count = len(p['configuration']['history'])
    values = np.asarray(total)
    if (values.shape != (count*28,) or values.dtype != np.float64
            or not np.isfinite(values).all() or (values < 0).any()
            or forecasted['families'].shape[1] != count):
        raise ValueError('Complete aligned finite forecast sum required')
    output = dict(forecast=(values.reshape(count,28)/6).tolist(),models=forecasted['models'],
                  horizon=28,model_families=6,scope=forecasted['scope'])
    json.dumps(output, allow_nan=False)
    return output

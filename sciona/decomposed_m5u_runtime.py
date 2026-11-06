"""Complete uncertainty lifecycle around independently bound numeric reduction."""
import hashlib
import json
import numpy as np
from sciona.m5u_execution import Prepared, prepare, unpack
from sciona.m5u_pipeline import run_unrestored
from sciona.m5u_training import QUANTILES, PARAMETERS


def checked_payload(prepared):
    if (type(prepared) is not Prepared or type(prepared.configuration) is not str
            or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint):
        raise ValueError('Unchanged prepared input required')
    payload=json.loads(prepared.configuration)
    if prepare(payload)!=prepared:
        raise ValueError('Canonical prepared input required')
    return payload


def fit_forecast(prepared):
    # Full default search, all levels and repeated queries; no speed overrides.
    return run_unrestored(**unpack(checked_payload(prepared)))


def checked_forecast(forecasted):
    if type(forecasted) is not dict:
        raise ValueError('Complete same-invocation forecasts required')
    values=np.asarray(forecasted['unadjusted'])
    levels=np.asarray(forecasted['levels']);factors=np.asarray(forecasted['factors'])
    if (values.ndim!=3 or values.shape[:2]!=(9,28) or not values.shape[2]
            or values.dtype!=np.float64 or not np.isfinite(values).all()
            or levels.shape!=(values.shape[2],) or levels.dtype.kind not in 'iu'
            or set(levels)!=set(PARAMETERS)
            or factors.shape!=levels.shape or factors.dtype.kind not in 'iuf'
            or not np.isfinite(factors).all() or (factors<=0).any()
            or tuple(forecasted['quantiles'])!=QUANTILES
            or type(forecasted['adjustment']) is not float or forecasted['adjustment']!=.7):
        raise ValueError('Complete finite 14-level, nine-quantile, 28-step forecasts required')
    for level in np.unique(levels):
        if np.unique(factors[levels==level]).size!=1:
            raise ValueError('One multiplier per level required')
    return values,levels,factors


def restoration_inputs(forecasted):
    values,levels,factors=checked_forecast(forecasted)
    native=values.astype(float,copy=True)/factors[None,None,:]
    selected=[level for level in np.unique(levels) if level<=9]
    totals=np.stack([native[QUANTILES.index(.5)][:,levels==level].sum(axis=1) for level in selected])
    if not np.isfinite(native).all() or not np.isfinite(totals).all() or (totals==0).any():
        raise ValueError('Finite native forecasts and nonzero median totals required')
    state=dict(native=native,levels=levels.copy(),selected=selected,totals=totals.copy())
    return state,totals,np.ones(len(selected))


def restore(state,total):
    if type(state) is not dict:
        raise ValueError('Same-invocation restoration state required')
    native=np.asarray(state['native']);levels=np.asarray(state['levels'])
    totals=np.asarray(state['totals']);selected=state['selected'];total=np.asarray(total)
    if (native.ndim!=3 or native.shape[:2]!=(9,28) or not native.shape[2]
            or native.dtype!=np.float64 or not np.isfinite(native).all()
            or levels.shape!=(native.shape[2],) or levels.dtype.kind not in 'iu'
            or set(levels)!=set(PARAMETERS) or list(selected)!=list(range(1,10))
            or totals.shape!=(9,28) or totals.dtype!=np.float64
            or not np.isfinite(totals).all() or (totals==0).any()
            or total.shape!=(28,) or total.dtype!=np.float64 or not np.isfinite(total).all()):
        raise ValueError('Complete finite restoration state and summed totals required')
    result=native.copy()
    target=total/len(selected)
    for level,level_total in zip(selected,totals):
        correction=(1-.7)+.7*target/level_total
        result[:,:,levels==level]*=correction[None,:,None]
    if not np.isfinite(result).all():
        raise ValueError('Hierarchy restoration overflow')
    return result


def result(forecasted,restored):
    values,_,_=checked_forecast(forecasted)
    restored=np.asarray(restored)
    if restored.shape!=values.shape or restored.dtype!=np.float64 or not np.isfinite(restored).all():
        raise ValueError('Complete restored forecast required')
    output=dict(forecast=restored.tolist(),levels=forecasted['levels'].tolist(),
        hierarchy_indices=forecasted['hierarchy_indices'].tolist(),quantiles=list(forecasted['quantiles']),
        horizon=28,reports=forecasted['reports'],normalization=forecasted['normalization'],scope=forecasted['scope'])
    json.dumps(output,allow_nan=False)
    return output

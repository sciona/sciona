"""Unpack the original finite JSON recipe without executing hidden stages."""
import copy
import json
import math


def unpack(payload):
    fields={'version','transactions','entities','boundaries','feature_controls','training_controls'}
    controls={'lags','rolling_windows','calendar_cycle','cold_start'}
    if not isinstance(payload,dict) or set(payload)!=fields:raise ValueError('Exact forecasting payload required')
    try:json.dumps(payload,allow_nan=False)
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON payload required') from None
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported payload version')
    if not isinstance(payload['feature_controls'],dict) or set(payload['feature_controls'])!=controls:
        raise ValueError('Exact feature controls required')
    if not isinstance(payload['training_controls'],dict):raise ValueError('Training controls required')
    p=copy.deepcopy(payload);c=p['feature_controls']
    if type(c['cold_start']) not in (int,float) or not math.isfinite(c['cold_start']) or not 0<=c['cold_start']<=20:
        raise ValueError('Bounded recipe cold-start mean required')
    return (p['transactions'],p['entities'],p['boundaries'],c['lags'],c['rolling_windows'],
            c['calendar_cycle'],float(c['cold_start']),p['training_controls'],0.,20.,[0,1,2])

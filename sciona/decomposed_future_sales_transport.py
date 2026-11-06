"""Finite JSON transport for isolated numerical search and refit operations."""
import json
import math
import os
import subprocess
import sys
import numpy as np
from sciona.future_sales_features import Panel


def validate_panel(panel,categorical_columns,lower,upper):
    if not isinstance(panel,Panel):raise ValueError('Panel required')
    x,y,t=panel.features,panel.targets,panel.periods
    if not isinstance(x,np.ndarray) or x.dtype!=np.float64 or x.ndim!=2 or min(x.shape)<1 or np.isinf(x).any():
        raise ValueError('Nonempty float64 features without infinities required')
    if not isinstance(y,np.ndarray) or y.dtype!=np.float64 or y.shape!=(len(x),):raise ValueError('Aligned float64 targets required')
    if not isinstance(t,np.ndarray) or t.dtype!=np.int64 or t.shape!=y.shape or (t<0).any():raise ValueError('Aligned int64 periods required')
    if type(panel.entity_count) is not int or panel.entity_count<1 or len(x)%panel.entity_count:
        raise ValueError('Complete rectangular panel required')
    expected=np.repeat(np.arange(len(x)//panel.entity_count),panel.entity_count)
    if not np.array_equal(t,expected) or t[-1]<2:raise ValueError('Ordered complete observed and forecast periods required')
    if not np.isfinite(y[t<t[-1]]).all() or not np.isnan(y[t==t[-1]]).all():
        raise ValueError('Finite observed targets and unknown forecast targets required')
    if not isinstance(panel.feature_names,tuple) or len(panel.feature_names)!=x.shape[1] or any(not isinstance(n,str) or not n for n in panel.feature_names) or len(set(panel.feature_names))!=len(panel.feature_names):
        raise ValueError('Unique feature labels required')
    if not isinstance(categorical_columns,list) or any(type(v) is not int or not 0<=v<x.shape[1] for v in categorical_columns) or len(set(categorical_columns))!=len(categorical_columns):
        raise ValueError('Distinct explicit categorical column indices required')
    categorical=x[:,categorical_columns]
    if not np.isfinite(categorical).all() or (categorical<0).any() or (categorical>=2**31).any() or not np.array_equal(categorical,np.floor(categorical)):
        raise ValueError('Finite nonnegative int32-compatible categorical values required')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (lower,upper)) or lower>=upper:
        raise ValueError('Finite ordered prediction bounds required')


def validate_selection(selection,config):
    fields={'parameters','trials','selected_trial','selected_rounds','training_rows','validation_rows','refit_rows','forecast_rows'}
    if not isinstance(selection,dict) or set(selection)!=fields:raise ValueError('Exact selection packet required')
    for key in fields-{'parameters','trials'}:
        if type(selection[key]) is not int or selection[key]<0:raise ValueError('Integer selection metadata required')
    if not 0<=selection['selected_trial']<config['trials'] or not 1<=selection['selected_rounds']<=config['max_rounds']:
        raise ValueError('Selected trial or iteration outside configured budget')
    if min(selection[k] for k in ('training_rows','validation_rows','refit_rows','forecast_rows'))<1:
        raise ValueError('Nonempty selection populations required')
    params=selection['parameters'];space=config['search_space']
    if not isinstance(params,dict) or set(params)!=set(space):raise ValueError('Exact selected parameters required')
    for name,value in params.items():
        if type(value) not in (int,float) or not math.isfinite(value) or not space[name][0]<=value<=space[name][1]:
            raise ValueError('Selected parameter outside configured search')
        if name!='learning_rate' and type(value) is not int:raise ValueError('Integer selected tree parameter required')
    trials=selection['trials']
    if not isinstance(trials,list) or len(trials)!=config['trials']:raise ValueError('Complete configured trial history required')
    for index,trial in enumerate(trials):
        if not isinstance(trial,dict) or set(trial)!={'trial','best_iteration','validation_rmse'}:raise ValueError('Exact trial record required')
        if type(trial['trial']) is not int or trial['trial']!=index or type(trial['best_iteration']) is not int or not 1<=trial['best_iteration']<=config['max_rounds']:
            raise ValueError('Ordered trials with valid stopping iterations required')
        value=trial['validation_rmse']
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative validation score required')
    winner=min(range(len(trials)),key=lambda i:trials[i]['validation_rmse'])
    if selection['selected_trial']!=winner or selection['selected_rounds']!=trials[winner]['best_iteration']:
        raise ValueError('Selection does not match best recorded trial')


def encode_panel(panel):
    def nullable(array):
        values=array.astype(object);values[np.isnan(array)]=None
        return values.tolist()
    return dict(features=nullable(panel.features),targets=nullable(panel.targets),periods=panel.periods.tolist(),
                entity_count=panel.entity_count,feature_names=list(panel.feature_names))


def decode_panel(raw):
    if not isinstance(raw,dict) or set(raw)!={'features','targets','periods','entity_count','feature_names'}:
        raise ValueError('Exact panel transport fields required')
    for key in ('features','targets','periods','feature_names'):
        if not isinstance(raw[key],list):raise ValueError('Panel arrays must be lists')
    if any(type(v) is not int for v in raw['periods']):raise ValueError('Integer period transport required')
    for rows in (raw['features'],[raw['targets']]):
        for row in rows:
            if not isinstance(row,list) or any(v is not None and (type(v) not in (int,float) or not math.isfinite(v)) for v in row):
                raise ValueError('Finite numeric or null panel transport required')
    return Panel(np.array(raw['features'],dtype=np.float64),np.array(raw['targets'],dtype=np.float64),
                 np.array(raw['periods'],dtype=np.int64),raw['entity_count'],tuple(raw['feature_names']))


def invoke(operation,panel,config,categorical_columns,lower,upper,selection=None):
    if operation not in ('search','refit'):raise ValueError('Unknown training operation')
    validate_panel(panel,categorical_columns,lower,upper)
    request=dict(operation=operation,panel=encode_panel(panel),config=config,
                 categorical_columns=categorical_columns,lower=lower,upper=upper,selection=selection)
    wire=json.dumps(request,allow_nan=False)
    try:
        timeout=float(os.environ.get('SCIONA_FUTURE_SALES_TIMEOUT_SECONDS','3600'))
        if not math.isfinite(timeout) or timeout<=0:raise ValueError()
    except (ValueError,TypeError):raise ValueError('Positive finite worker timeout required') from None
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        result=subprocess.run([sys.executable,'-m','sciona.decomposed_future_sales_worker'],input=wire,
            text=True,capture_output=True,timeout=timeout,env=env,check=False)
    except (OSError,subprocess.TimeoutExpired):raise RuntimeError('Isolated training failed or timed out') from None
    if result.returncode:raise RuntimeError('Isolated training failed')
    try:
        response=json.loads(result.stdout)
        json.dumps(response,allow_nan=False)
        if operation=='search':validate_selection(response,config)
        else:
            if not isinstance(response,dict) or set(response)!=(set(selection)-{'parameters'})|{'predictions'}:
                raise ValueError('Invalid refit fields')
            if any(response[k]!=v for k,v in selection.items() if k!='parameters'):raise ValueError('Refit metadata differs')
            validate_selection(dict(parameters=selection['parameters'],**{k:v for k,v in response.items() if k!='predictions'}),config)
            predictions=response['predictions']
            if not isinstance(predictions,list) or len(predictions)!=panel.entity_count or any(type(v) not in (int,float) or not math.isfinite(v) or not lower<=v<=upper for v in predictions):
                raise ValueError('Invalid refit predictions')
        expected_counts=dict(training_rows=int(((panel.periods>=config['train_start'])&(panel.periods<config['validation_start'])).sum()),
            validation_rows=int(((panel.periods>=config['validation_start'])&(panel.periods<panel.periods[-1])).sum()),
            refit_rows=int(((panel.periods>=config['train_start'])&(panel.periods<panel.periods[-1])).sum()),forecast_rows=panel.entity_count)
        if any(response[k]!=v for k,v in expected_counts.items()):raise ValueError('Worker population counts differ')
    except (ValueError,TypeError,KeyError):raise RuntimeError('Invalid isolated training response') from None
    return response


def search(panel,config,categorical_columns,lower,upper):
    return invoke('search',panel,config,categorical_columns,lower,upper)


def refit(panel,config,categorical_columns,lower,upper,selection):
    validate_selection(selection,config)
    return invoke('refit',panel,config,categorical_columns,lower,upper,selection)

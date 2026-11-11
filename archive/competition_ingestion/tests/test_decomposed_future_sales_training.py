"""Complete isolated stage parity and reusable numeric training contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from sciona.future_sales_features import build_panel,Panel
from sciona.future_sales_contract import prepare,execute
from sciona import decomposed_future_sales_transport as transport
from scripts.future_sales_synthetic import payload


def inputs():
    p=payload()
    panel=build_panel(p['transactions'],p['entities'],p['boundaries'],**p['feature_controls'])
    return p,panel


@pytest.mark.parametrize('seed,trials',[(92,3),(17,4),(61,5)])
def test_full_split_lifecycle_equals_reference(seed,trials):
    p,panel=inputs();p['training_controls'].update(seed=seed,trials=trials)
    c=p['training_controls']
    selected=transport.search(panel,c,[0,1,2],0.,20.)
    actual=transport.refit(panel,c,[0,1,2],0.,20.,selected)
    expected=execute(prepare(p))
    assert actual==expected
    assert len(selected['trials'])==trials


@pytest.mark.parametrize('width,categories',[(1,[]),(6,[0,2])])
def test_numeric_reuse_with_explicit_category_and_bound_policy(width,categories):
    rng=np.random.default_rng(5)
    x=rng.normal(size=(18,width)).astype(np.float64)
    for col in categories:x[:,col]=np.arange(18)%3
    y=np.linspace(-4,4,18);y[-3:]=np.nan
    panel=Panel(x,y,np.repeat(np.arange(6),3),3,tuple('feature_'+str(i) for i in range(width)))
    c=payload()['training_controls'];c.update(train_start=0,validation_start=3)
    selected=transport.search(panel,c,categories,-5.,5.)
    result=transport.refit(panel,c,categories,-5.,5.,selected)
    assert (result['training_rows'],result['validation_rows'],result['refit_rows'],result['forecast_rows'])==(9,6,15,3)
    assert len(result['predictions'])==3 and all(-5<=v<=5 for v in result['predictions'])


def test_panel_roundtrip_preserves_float_bytes():
    _,panel=inputs()
    raw=json.loads(json.dumps(transport.encode_panel(panel),allow_nan=False))
    restored=transport.decode_panel(raw)
    for key in ('features','targets','periods'):assert getattr(restored,key).tobytes()==getattr(panel,key).tobytes()


@pytest.mark.parametrize('fault',['observed_nan','future_known','period_order','fractional_category','duplicate_category','nan_bound','ragged_population'])
def test_invalid_panel_fails_before_process(monkeypatch,fault):
    _,panel=inputs();categories=[0,1,2];lower=0.
    if fault=='observed_nan':panel.targets[0]=np.nan
    elif fault=='future_known':panel.targets[-1]=0
    elif fault=='period_order':panel.periods[0]=1
    elif fault=='fractional_category':panel.features[0,0]=.5
    elif fault=='duplicate_category':categories=[0,0]
    elif fault=='nan_bound':lower=np.nan
    else:panel=Panel(panel.features,panel.targets,panel.periods,7,panel.feature_names)
    def forbidden(*args,**kwargs):raise AssertionError('Invalid panel spawned a worker')
    monkeypatch.setattr(transport.subprocess,'run',forbidden)
    with pytest.raises(ValueError):transport.search(panel,payload()['training_controls'],categories,lower,20.)


@pytest.mark.parametrize('fault',['failure','timeout','bad_json','wrong_counts','bool_count','missing_trial','invalid_winner'])
def test_worker_response_guards(monkeypatch,fault):
    p,panel=inputs();config=p['training_controls']
    response=dict(parameters=dict(num_leaves=3,min_data_in_leaf=2,learning_rate=.1),
        trials=[dict(trial=i,best_iteration=2,validation_rmse=float(i)) for i in range(3)],
        selected_trial=0,selected_rounds=2,training_rows=40,validation_rows=16,refit_rows=56,forecast_rows=8)
    if fault=='wrong_counts':response['training_rows']=41
    elif fault=='bool_count':response['selected_trial']=False
    elif fault=='missing_trial':response['trials'].pop()
    elif fault=='invalid_winner':response['selected_trial']=1
    def simulated(*args,**kwargs):
        assert all(kwargs['env'][name]=='1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'))
        if fault=='timeout':raise subprocess.TimeoutExpired('worker',1,output='private sentinel')
        return SimpleNamespace(returncode=1 if fault=='failure' else 0,
            stdout='private sentinel' if fault=='bad_json' else json.dumps(response),stderr='private sentinel')
    monkeypatch.setattr(transport.subprocess,'run',simulated)
    with pytest.raises(RuntimeError) as error:transport.search(panel,config,[0,1,2],0.,20.)
    assert 'private sentinel' not in str(error.value)


def test_bad_numeric_transport_rejected():
    _,panel=inputs();raw=transport.encode_panel(panel)
    raw['features'][0][0]=True
    with pytest.raises(ValueError):transport.decode_panel(raw)


def test_stage_fit_budgets_and_model_based_tpe():
    code='''
from scripts.future_sales_synthetic import payload
from sciona.future_sales_features import build_panel
from sciona.decomposed_future_sales_training import search,refit
import lightgbm as lgb
from optuna.samplers import TPESampler
p=payload();c=p['training_controls'];panel=build_panel(p['transactions'],p['entities'],p['boundaries'],**p['feature_controls'])
calls=[];sampled=[];original=lgb.train;sample=TPESampler._sample
def tracked(parameters,data,*args,**kwargs):
    calls.append(len(data.get_label()))
    return original(parameters,data,*args,**kwargs)
def tracked_sample(self,*args,**kwargs):
    sampled.append(1)
    return sample(self,*args,**kwargs)
lgb.train=tracked;TPESampler._sample=tracked_sample
selected=search(panel,c,[0,1,2],0.,20.)
if calls!=[40]*c['trials'] or not sampled:raise RuntimeError('Search lifecycle differs')
calls.clear();sampled.clear()
refit(panel,c,[0,1,2],0.,20.,selected)
if calls!=[56] or sampled:raise RuntimeError('Final refit lifecycle differs')
print('stage budgets verified')
'''
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='stage budgets verified'

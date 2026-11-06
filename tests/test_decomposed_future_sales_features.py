"""Synthetic parity, causality and numeric reuse for extracted feature stages."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from sciona.atoms.ml.tabular.gradient_boosting.atoms import rolling_statistics
from sciona.future_sales_features import build_panel
from sciona import decomposed_future_sales_features as stages
from scripts.future_sales_synthetic import payload


def extracted(p):
    controls=p['feature_controls']
    totals,groups=stages.aggregate(p['transactions'],p['entities'],p['boundaries'])
    targets=stages.bounded_targets(totals,0.,20.)
    lagged=stages.lag_features(targets,controls['lags'],controls['rolling_windows'],rolling_statistics)
    encoded=stages.encoding_features(targets,groups,controls['calendar_cycle'],controls['cold_start'])
    return stages.assemble(targets,lagged,encoded,controls['lags'],controls['rolling_windows'])


@pytest.mark.parametrize('seed',range(12))
def test_complete_reference_parity(seed):
    p=payload();rng=np.random.default_rng(seed)
    for row in p['transactions']:row['quantity']=float(rng.uniform(-15,35))
    p['feature_controls'].update(lags=[1,3,12],rolling_windows=[2,4,12],calendar_cycle=seed+1,cold_start=seed/2)
    before=copy.deepcopy(p)
    actual=extracted(p)
    expected=build_panel(p['transactions'],p['entities'],p['boundaries'],**p['feature_controls'])
    for name in ('features','targets','periods'):
        a,b=getattr(actual,name),getattr(expected,name)
        assert np.array_equal(a,b,equal_nan=True)
        assert a.tobytes()==b.tobytes()
    assert actual.feature_names==expected.feature_names and actual.entity_count==expected.entity_count
    assert p==before


def test_net_compensated_sum_before_clip():
    p=payload();p['transactions']=[]
    for q in [1e16,1.,-1e16]:p['transactions'].append(dict(day=1,store=0,item=0,quantity=q))
    totals,_=stages.aggregate(p['transactions'],p['entities'],p['boundaries'])
    assert totals[0,0]==1.
    assert stages.bounded_targets(totals,0.,20.)[0,0]==1.


@pytest.mark.parametrize('window',[8,16,33,64])
def test_long_window_reference_bytes(window):
    p=payload();rng=np.random.default_rng(window)
    p['boundaries']=list(range(72))
    p['transactions']=[dict(day=period,store=e['store'],item=e['item'],quantity=float(rng.uniform(0,20)))
                       for period in range(70) for e in p['entities']]
    p['feature_controls'].update(lags=[1,7,30],rolling_windows=[window])
    actual=extracted(p)
    expected=build_panel(p['transactions'],p['entities'],p['boundaries'],**p['feature_controls'])
    assert actual.features.tobytes()==expected.features.tobytes()


@pytest.mark.parametrize('group_width',[1,2,5])
def test_generic_group_width_and_prior_only(group_width):
    targets=np.arange(24,dtype=np.float64).reshape(6,4);targets[-1]=np.nan
    groups=np.tile(np.arange(group_width,dtype=np.int64),(4,1));groups[2:]+=1
    baseline=stages.encoding_features(targets,groups,7,-4.)
    assert baseline.shape==(24,2*group_width+5)
    changed=targets.copy();changed[3:5]+=100
    revised=stages.encoding_features(changed,groups,7,-4.)
    assert np.array_equal(baseline[:16],revised[:16])
    assert not np.array_equal(baseline[16:],revised[16:])


def test_lags_are_prior_only_with_full_windows():
    values=np.array([[2.],[4.],[8.],[16.],[np.nan]])
    result=stages.lag_features(values,[1],[2],rolling_statistics)
    assert np.array_equal(result[:,1],[np.nan,np.nan,3.,6.,12.],equal_nan=True)
    assert np.array_equal(result[:,0],[np.nan,2.,4.,8.,16.],equal_nan=True)


@pytest.mark.parametrize('case',['known_future','unknown_observed','dtype','empty','bad_groups','bad_cycle','bad_window','bad_callable'])
def test_invalid_intermediates(case):
    y=np.array([[1.,2.],[3.,4.],[np.nan,np.nan]])
    groups=np.array([[0],[1]],dtype=np.int64)
    with pytest.raises(ValueError):
        if case=='known_future':y[-1]=0;stages.encoding_features(y,groups,2,0.)
        elif case=='unknown_observed':y[0,0]=np.nan;stages.encoding_features(y,groups,2,0.)
        elif case=='dtype':stages.encoding_features(y.astype(np.float32),groups,2,0.)
        elif case=='empty':stages.bounded_targets(np.empty((0,2)),0.,20.)
        elif case=='bad_groups':stages.encoding_features(y,groups.astype(float),2,0.)
        elif case=='bad_cycle':stages.encoding_features(y,groups,True,0.)
        elif case=='bad_window':stages.lag_features(y,[1],[True],rolling_statistics)
        else:stages.lag_features(y,[1],[2],lambda *args:np.zeros((1,1)))


@pytest.mark.parametrize('seed,trials',[(92,3),(17,4),(61,5)])
def test_complete_training_parity_in_isolated_process(seed,trials):
    # Both full lifecycles execute in a fresh process; no model or input files.
    script='''
import sys
from scripts.future_sales_synthetic import payload
from sciona.future_sales_worker import run
from sciona.future_sales_training import train_forecast
from sciona import decomposed_future_sales_features as s
from sciona.atoms.ml.tabular.gradient_boosting.atoms import rolling_statistics
p=payload();p['training_controls'].update(seed=int(sys.argv[1]),trials=int(sys.argv[2]))
for i,row in enumerate(p['transactions']):row['quantity']=((i*7+int(sys.argv[1]))%31)-4
c=p['feature_controls']
totals,groups=s.aggregate(p['transactions'],p['entities'],p['boundaries'])
y=s.bounded_targets(totals,0.,20.)
lagged=s.lag_features(y,c['lags'],c['rolling_windows'],rolling_statistics)
encoded=s.encoding_features(y,groups,c['calendar_cycle'],c['cold_start'])
panel=s.assemble(y,lagged,encoded,c['lags'],c['rolling_windows'])
actual=train_forecast(panel,p['training_controls']);actual['predictions']=actual['predictions'].tolist()
expected=run(p)
if actual!=expected:raise RuntimeError('Complete extracted lifecycle differs')
if len(actual['trials'])!=int(sys.argv[2]):raise RuntimeError('Incomplete configured search')
print('full lifecycle equal')
'''
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-c',script,str(seed),str(trials)],
        cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0, result.stderr
    assert result.stdout.strip()=='full lifecycle equal'

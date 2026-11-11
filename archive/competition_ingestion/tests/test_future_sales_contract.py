import json
import pytest
from scripts.future_sales_synthetic import payload
from sciona.future_sales_contract import prepare,execute


def test_json_copy_and_execution():
    p=json.loads(json.dumps(payload(),allow_nan=False));prepared=prepare(p)
    p['transactions'][0]['quantity']=999
    assert prepared.payload['transactions'][0]['quantity']==0
    result=execute(prepared);assert result['forecast_rows']==8
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('case',['extra','version','nonfinite','unknown_entity','missing_feature','future_transaction'])
def test_boundary_failures(case):
    p=payload()
    if case=='extra':p['extra']=1
    elif case=='version':p['version']=True
    elif case=='nonfinite':p['transactions'][0]['quantity']=float('nan')
    elif case=='unknown_entity':p['transactions'][0]['store']=999
    elif case=='missing_feature':p['feature_controls'].pop('lags')
    else:p['transactions'][0]['day']=90
    with pytest.raises(ValueError):prepare(p)


def test_invalid_mutated_controls_fail_before_training(monkeypatch):
    import lightgbm
    def forbidden(*args,**kwargs):raise AssertionError('Invalid controls reached training')
    monkeypatch.setattr(lightgbm,'train',forbidden)
    p=prepare(payload());p.payload['training_controls']['trials']=1
    from sciona.future_sales_worker import run
    with pytest.raises(ValueError):run(p.payload)


def test_isolated_worker_matches_direct_execution():
    from sciona.future_sales_worker import run
    p=payload()
    assert execute(prepare(p))==run(p)


@pytest.mark.parametrize('case',['failure','timeout','malformed'])
def test_worker_failures_do_not_expose_private_output(monkeypatch,case):
    import subprocess
    from types import SimpleNamespace
    def failed(*args,**kwargs):
        if case=='timeout':raise subprocess.TimeoutExpired('worker',1,output='private sentinel')
        return SimpleNamespace(returncode=1 if case=='failure' else 0,stdout='private sentinel',stderr='private sentinel')
    monkeypatch.setattr(subprocess,'run',failed)
    with pytest.raises(RuntimeError) as error:execute(prepare(payload()))
    assert 'private sentinel' not in str(error.value)

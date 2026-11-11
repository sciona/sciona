import copy
import numpy as np
import pytest
import torch
from scripts.physical_operator_synthetic import payload
from sciona.physical_operator_state import prepare,tensors
from sciona.physical_operator_training import fit


def test_training_improves_and_restores_best():
    p=payload();before=torch.random.get_rng_state().clone();f=fit(p)
    assert torch.equal(before,torch.random.get_rng_state())
    assert len(f.history)==61
    assert f.history[-1]['training_mse'] < f.history[0]['training_mse']*.1
    assert f.validation_mse==min(r['validation_mse'] for r in f.history)
    assert f.best_epoch==min(f.history,key=lambda r:r['validation_mse'])['epoch']
    x,t=tensors(p['validation'],p['length'])
    with torch.no_grad():loss=float((f.model(x,t)-torch.tensor(p['validation']['targets'])).square().mean())
    assert loss==pytest.approx(f.validation_mse,rel=1e-5,abs=1e-10)


def test_validation_targets_do_not_change_training_trajectory():
    p=payload();p['controls']['epochs']=4
    first=fit(p)
    for row in p['validation']['targets']:row.reverse()
    second=fit(p)
    assert [r['training_mse'] for r in first.history]==[r['training_mse'] for r in second.history]
    assert [r['validation_mse'] for r in first.history]!=[r['validation_mse'] for r in second.history]


def test_repeat_and_query_independence():
    p=payload();p['controls']['epochs']=3
    first=fit(p)
    p['query']['states']=[[v*2 for v in row] for row in p['query']['states']]
    second=fit(p)
    assert first.history==second.history
    for k,v in first.model.state_dict().items():torch.testing.assert_close(v,second.model.state_dict()[k],rtol=0,atol=0)


def test_si_nondimensionalization():
    p=payload();first=prepare(p)
    p['length']=2.;p['grid']=[2*x for x in p['grid']]
    for name in ('training','validation','query'):p[name]['diffusivity']=[4*d for d in p[name]['diffusivity']]
    second=prepare(p)
    torch.testing.assert_close(tensors(first['training'],1.)[1],tensors(second['training'],2.)[1])

@pytest.mark.parametrize('mutate',[
    lambda p:p.update(version=True),
    lambda p:p['units'].update(time='ms'),
    lambda p:p['grid'].__setitem__(-1,1.),
    lambda p:p['training']['states'][0].__setitem__(0,-1.),
    lambda p:p['training']['targets'][0].__setitem__(0,10.),
    lambda p:p['validation']['groups'].__setitem__(0,'synthetic0'),
    lambda p:p['query']['elapsed'].pop(),
    lambda p:p['controls'].update(modes=10),
    lambda p:p['query']['states'][0].__setitem__(0,True),
])
def test_invalid_physical_contract(mutate):
    p=payload();mutate(p)
    with pytest.raises(ValueError):prepare(p)

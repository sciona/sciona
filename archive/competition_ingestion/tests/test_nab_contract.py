import copy
import json
import pytest
from scripts.nab_synthetic import payload
from sciona.nab_contract import prepare,execute


def test_private_copy_and_label_independence():
    p=payload();p['values']=p['values'][:25];p['windows']=[[18,20]]
    prepared=prepare(json.loads(json.dumps(p,allow_nan=False)))
    p['values'][0]=999
    first=execute(prepared)
    changed=copy.deepcopy(prepared.payload);changed['windows']=[]
    second=execute(prepare(changed))
    assert first['detection']==second['detection']
    assert first['evaluation']['normalization_defined'] and not second['evaluation']['normalization_defined']


@pytest.mark.parametrize('case',['version','extra','bool_value','nan','overlap','negative_seed','zero_trees','bad_cost'])
def test_invalid_inputs(case):
    p=payload()
    if case=='version':p['version']=True
    elif case=='extra':p['extra']=None
    elif case=='bool_value':p['values'][0]=True
    elif case=='nan':p['values'][0]=float('nan')
    elif case=='overlap':p['windows']=[[4,8],[7,10]]
    elif case=='negative_seed':p['detector']['seed']=-1
    elif case=='zero_trees':p['detector']['n_estimators']=0
    elif case=='bad_cost':p['costs']['tpWeight']=-1
    with pytest.raises(ValueError):prepare(p)


def test_mutated_prepared_revalidated():
    p=prepare(payload());p.payload['detector']['seed']=-1
    with pytest.raises(ValueError):execute(p)

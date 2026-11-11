import copy
import json
from unittest.mock import patch
import pytest
from scripts.openvaccine_synthetic import payload
from sciona.openvaccine_contract import prepare,execute


def test_json_and_copy():
    p=json.loads(json.dumps(payload(),allow_nan=False));prepared=prepare(p)
    p['targets'][0][0][0]=999
    assert prepared.payload['targets'][0][0][0]!=999


@pytest.mark.parametrize('case', ['extra','version','seed','nan','string','bool_cluster','duplicate','missing','overlap','late_round','steps'])
def test_invalid_payload(case):
    p=payload()
    if case=='extra':p['unexpected']=1
    elif case=='version':p['version']=True
    elif case=='seed':p['seed']=-1
    elif case=='nan':p['targets'][0][0][0]=float('nan')
    elif case=='string':p['targets'][0][0][0]='1'
    elif case=='bool_cluster':p['cluster_ids'][0]=True
    elif case=='duplicate':p['splits'][1]=copy.deepcopy(p['splits'][0])
    elif case=='missing':p['splits'].pop()
    elif case=='overlap':p['splits'][0]['validation']=[0,2]
    elif case=='late_round':
        p['pseudo_rounds'].append(copy.deepcopy(p['pseudo_rounds'][0]))
        p['pseudo_rounds'][1][-1]['sample_weights']=[0.,0.]
    elif case=='steps':p['pretraining_steps']=0
    with pytest.raises(ValueError):prepare(p)


def test_execute_revalidates_mutated_prepared():
    prepared=prepare(payload());prepared.payload['seed']=-1
    with patch.dict('os.environ',{},clear=True),pytest.raises(ValueError,match='seed'):
        execute(prepared)

"""Synthetic adversarial checks for the public execution boundary."""
import copy
import json
import pytest
from scripts.tabular_ensemble_synthetic import payload
from sciona import tabular_ensemble_contract as contract

def test_complete_boundary():
    source = payload(); prepared = contract.prepare(source)
    source['training']['numeric'][0][0] = -999
    assert prepared.payload['training']['numeric'][0][0] == 0
    result = contract.execute(prepared)
    json.dumps(result,allow_nan=False)
    assert (result['training_rows'],result['calibration_rows'],result['query_rows']) == (12,4,2)
    assert (result['folds'],result['oof_rows'],result['models'],result['forest_trees']) == (3,12,2,64)
    assert len(result['probabilities']) == 2
    assert all(0 <= p <= 1 for p in result['probabilities'])
    assert result['classes'] == [int(p >= .5) for p in result['probabilities']]

@pytest.mark.parametrize('mutation',[
    lambda p:p.update(version=True),
    lambda p:p.update(extra=1),
    lambda p:p['controls'].update(trees=True),
    lambda p:p['controls'].update(clip_low=True),
    lambda p:p['query']['numeric'][0].__setitem__(0,float('nan')),
    lambda p:p['query'].update(groups=('query0','query1')),
    lambda p:p['query']['groups'].__setitem__(0,'train0'),
    lambda p:p['query']['groups'].__setitem__(0,'cal0'),
    lambda p:p['calibration']['groups'].__setitem__(0,'train0'),
    lambda p:p['query']['numeric'][0].append(1),
    lambda p:p['training']['folds'].__setitem__(0,1),
    lambda p:p['training'].update(folds=[0]*12),
    lambda p:p['training'].update(labels=[0]*4+[1]*8),
])
def test_invalid_payload_rejected_before_fit(mutation,monkeypatch):
    p=payload(); mutation(p)
    def forbidden(*args,**kwargs): raise AssertionError('Must reject before fit')
    monkeypatch.setattr(contract,'fit_stacker',forbidden)
    with pytest.raises(ValueError): contract.execute(contract.Prepared(p))

def test_mutable_prepared_revalidated(monkeypatch):
    prepared=contract.prepare(payload())
    prepared.payload['query']['groups'][0]='train0'
    def forbidden(*args,**kwargs): raise AssertionError('Must reject before fit')
    monkeypatch.setattr(contract,'fit_stacker',forbidden)
    with pytest.raises(ValueError): contract.execute(prepared)

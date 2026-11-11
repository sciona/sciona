import json
import numpy as np
import pytest
from sklearn.metrics import f1_score
from scripts.biosignal_sequence_synthetic import payload
from sciona import biosignal_sequence_contract as contract


def test_complete_and_calibration_isolation():
    p=payload();first=contract.execute(contract.prepare(p));json.dumps(first,allow_nan=False)
    assert (first['training_rows'],first['calibration_rows'],first['query_rows'])==(4,4,2)
    assert first['training_windows']==18 and first['query_windows']==7
    assert first['final_training_loss']<first['initial_training_loss']*.25
    p['calibration']['labels']=[1-v for v in p['calibration']['labels']]
    second=contract.execute(contract.prepare(p))
    assert first['scores']==second['scores'] and first['final_training_loss']==second['final_training_loss']
    assert first['threshold']!=second['threshold']


def test_independent_threshold_search():
    scores=np.array([.1,.3,.4,.9]);labels=[1,0,0,1]
    candidates=list(scores)+[np.nextafter(scores.max(),np.inf)]
    expected=max(candidates,key=lambda c:(f1_score(labels,scores>=c),c))
    assert contract.threshold(scores,labels)==expected==.9

@pytest.mark.parametrize('mutate',[
    lambda p:p.update(version=True),
    lambda p:p['query']['subjects'].__setitem__(0,'train0'),
    lambda p:p['query']['subjects'].__setitem__(0,'cal0'),
    lambda p:p['calibration']['subjects'].__setitem__(0,'train0'),
    lambda p:p['query'].update(labels=[0,1]),
    lambda p:p['controls'].update(sample_rate=True),
    lambda p:p['controls'].update(max_time=3),
    lambda p:p['controls'].update(stride=33),
    lambda p:p['query']['signals'][0].append([0.]*64),
    lambda p:p['training'].update(labels=[0]*4),
])
def test_invalid_before_fit(mutate,monkeypatch):
    p=payload();mutate(p)
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before fit')
    monkeypatch.setattr(contract,'fit',forbidden)
    with pytest.raises(ValueError):contract.execute(contract.Prepared(p))


def test_prepared_copy_and_mutation():
    p=payload();prepared=contract.prepare(p);p['version']=2
    assert prepared.payload['version']==1
    prepared.payload['version']=False
    with pytest.raises(ValueError):contract.execute(prepared)

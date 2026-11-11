import json
import numpy as np
import pytest
from sklearn.metrics import f1_score
from scripts.image_tabular_synthetic import payload
from sciona import image_tabular_contract as contract


def test_complete_and_calibration_isolation():
    p=payload();first=contract.execute(contract.prepare(p));json.dumps(first,allow_nan=False)
    assert (first['training_rows'],first['calibration_rows'],first['query_rows'])==(12,4,2)
    assert first['folds']==3 and first['oof_rows']==12 and first['visual_features']==31
    assert first['final_training_loss']<first['initial_training_loss']*.5
    p['calibration']['labels']=[1-v for v in p['calibration']['labels']]
    second=contract.execute(contract.prepare(p))
    assert first['scores']==second['scores'] and first['final_training_loss']==second['final_training_loss']
    assert first['threshold']!=second['threshold']


def test_independent_f1():
    scores=np.array([.1,.3,.4,.9]);labels=[1,0,0,1]
    candidates=list(scores)+[np.nextafter(scores.max(),np.inf)]
    assert contract.threshold(scores,labels)==max(candidates,key=lambda t:(f1_score(labels,scores>=t),t))==.9

@pytest.mark.parametrize('mutate',[
    lambda p:p.update(version=True),
    lambda p:p['query']['groups'].__setitem__(0,'train0'),
    lambda p:p['query']['groups'].__setitem__(0,'cal0'),
    lambda p:p['calibration']['groups'].__setitem__(0,'train0'),
    lambda p:p['query'].update(labels=[0,1]),
    lambda p:p['controls'].update(dropout=True),
    lambda p:p['query']['numeric'][0].append(1.),
    lambda p:p['training']['folds'].__setitem__(0,1),
])
def test_invalid_before_fit(mutate,monkeypatch):
    p=payload();mutate(p)
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before fit')
    monkeypatch.setattr(contract,'fit',forbidden)
    with pytest.raises(ValueError):contract.execute(contract.Prepared(p))


def test_prepared_copy_and_revalidation():
    p=payload();prepared=contract.prepare(p);p['version']=2
    assert prepared.payload['version']==1
    prepared.payload['version']=False
    with pytest.raises(ValueError):contract.execute(prepared)

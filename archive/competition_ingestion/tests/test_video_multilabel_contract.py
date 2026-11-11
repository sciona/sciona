import json
import numpy as np
import pytest
from scripts.video_multilabel_synthetic import payload
from sciona import video_multilabel_contract as contract


def test_complete_lifecycle_and_calibration_isolation():
    p=payload();first=contract.execute(contract.prepare(p));json.dumps(first,allow_nan=False)
    assert (first['training_rows'],first['calibration_rows'],first['query_rows'])==(8,4,2)
    assert first['sparse_models']==first['label_count']==2
    assert first['final_training_loss']<first['initial_training_loss']*.25
    assert first['labels']==(np.asarray(first['scores'])>=first['thresholds']).astype(int).tolist()
    p['calibration']['labels']=[[1-v for v in row] for row in p['calibration']['labels']]
    second=contract.execute(contract.prepare(p))
    assert first['scores']==second['scores'] and first['final_training_loss']==second['final_training_loss']
    assert first['thresholds']!=second['thresholds']

@pytest.mark.parametrize('mutate',[
    lambda p:p.update(version=True),
    lambda p:p['calibration']['groups'].__setitem__(0,'train0'),
    lambda p:p['query']['groups'].__setitem__(0,'train0'),
    lambda p:p['query']['groups'].__setitem__(0,'cal0'),
    lambda p:p['query'].update(labels=[[0,0],[1,1]]),
    lambda p:p['controls'].update(batch_size=True),
    lambda p:p['query']['videos'][0][0].append(1.),
    lambda p:p['training'].update(sparse=[{} for _ in range(8)]),
    lambda p:p['calibration']['labels'].__setitem__(0,[0]),
])
def test_invalid_before_fit(mutate,monkeypatch):
    p=payload();mutate(p)
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before fit')
    monkeypatch.setattr(contract,'fit',forbidden)
    with pytest.raises(ValueError):contract.execute(contract.Prepared(p))


def test_prepared_copy_revalidated():
    p=payload();prepared=contract.prepare(p);p['version']=7
    assert prepared.payload['version']==1
    prepared.payload['version']=False
    with pytest.raises(ValueError):contract.execute(prepared)

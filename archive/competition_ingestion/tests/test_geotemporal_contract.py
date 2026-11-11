import json
import numpy as np
import pytest
from scripts.geotemporal_synthetic import payload
from sciona import geotemporal_contract as contract


def test_complete_lifecycle():
    result=contract.execute(contract.prepare(payload()));json.dumps(result,allow_nan=False)
    assert (result['training_rows'],result['validation_rows'],result['warmup_rows'],result['query_rows'])==(9,6,3,2)
    assert (result['folds'],result['trees'],result['feature_count'])==(2,32,11)
    assert len(result['predictions'])==2 and all(v>=0 for v in result['predictions'])
    assert np.isfinite(result['validation_mse'])

@pytest.mark.parametrize('mutate',[
    lambda p:p.update(version=True),
    lambda p:p['reference'].update(coordinates='latitude_longitude'),
    lambda p:p['reference'].update(distance_unit='km'),
    lambda p:p['reference'].update(shared_origin=1),
    lambda p:p['query'][0].update(target=1.),
    lambda p:p['query'][0].update(time=8.),
    lambda p:p['query'].append(dict(p['query'][0])),
    lambda p:p['controls'].update(trees=True),
    lambda p:p['blocks'].__setitem__(0,2),
])
def test_invalid_before_fit(mutate,monkeypatch):
    p=payload();mutate(p)
    def forbidden(*args,**kwargs):raise AssertionError('Must reject before fit')
    monkeypatch.setattr(contract,'fit',forbidden)
    with pytest.raises(ValueError):contract.execute(contract.Prepared(p))


def test_prepared_copy_revalidated():
    p=payload();prepared=contract.prepare(p);p['version']=2
    assert prepared.payload['version']==1
    prepared.payload['query'][0]['time']=0
    with pytest.raises(ValueError):contract.execute(prepared)

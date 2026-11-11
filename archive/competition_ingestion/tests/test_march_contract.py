import json
import pytest
from scripts.march_synthetic import payload
from sciona.march_contract import prepare,execute


def test_json_roundtrip_and_private_copy():
    p=json.loads(json.dumps(payload(),allow_nan=False));prepared=prepare(p)
    p['games'][0]['a_box']['points']=999
    assert prepared.payload['games'][0]['a_box']['points']==80
    result=execute(prepared)
    assert result['predicted_matchups']==2
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('case',['version','extra','missing_control','nan','empty'])
def test_boundary_failures(case):
    p=payload()
    if case=='version':p['version']=True
    elif case=='extra':p['extra']=1
    elif case=='missing_control':p['feature_controls'].pop('elo_k')
    elif case=='nan':p['regularization']=float('nan')
    else:p['prediction']=[]
    with pytest.raises(ValueError):prepare(p)


def test_revalidation_before_model_fit(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    def forbidden(*args,**kwargs):raise AssertionError('Invalid chronology reached model fit')
    monkeypatch.setattr(LogisticRegression,'fit',forbidden)
    p=prepare(payload());p.payload['calibration'][0]['season']=1
    with pytest.raises(ValueError):execute(p)

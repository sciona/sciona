"""Synthetic checks for the explicit recipe boundary."""
import pytest
from scripts.future_sales_synthetic import payload
from sciona.decomposed_future_sales_contract import unpack


def test_private_copy_and_explicit_policy():
    p=payload();out=unpack(p)
    assert len(out)==11
    assert out[-3:]==(0.,20.,[0,1,2])
    p['transactions'][0]['quantity']=12345
    p['training_controls']['trials']=999
    assert out[0][0]['quantity']==0
    assert out[7]['trials']==3


@pytest.mark.parametrize('fault',['extra','version','nonfinite','missing_controls','cold_bool','cold_low','cold_high','training'])
def test_invalid_boundary(fault):
    p=payload()
    if fault=='extra':p['extra']=1
    elif fault=='version':p['version']=True
    elif fault=='nonfinite':p['transactions'][0]['quantity']=float('inf')
    elif fault=='missing_controls':p['feature_controls'].pop('lags')
    elif fault=='cold_bool':p['feature_controls']['cold_start']=True
    elif fault=='cold_low':p['feature_controls']['cold_start']=-1
    elif fault=='cold_high':p['feature_controls']['cold_start']=21
    else:p['training_controls']=[]
    with pytest.raises(ValueError):unpack(p)

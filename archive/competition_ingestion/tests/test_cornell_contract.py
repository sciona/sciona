"""Synthetic Cornell graph boundary rejection and ownership checks."""
import copy
import numpy as np
import pytest
from scripts.cornell_synthetic import payload
from sciona.cornell_contract import prepare,execute


@pytest.mark.parametrize('change',[{'version':True},{'epochs':1},{'epochs':2.5},{'batch_size':3},
    {'seed':-1},{'initialization':'automatic'},{'source_path':'untrusted'},
    {'background':[]},{'short_noises':[]},{'background':[np.zeros(3200)]},
    {'inference':{'sample_rate':16000,'waveform':[1,2]}}])
def test_invalid_fields(change):
    value=payload(3200);value.update(change)
    with pytest.raises(ValueError):prepare(value)


@pytest.mark.parametrize('kind',['missing','duplicate','overlap','invalid_label'])
def test_invalid_folds(kind):
    value=payload(3200)
    if kind=='missing':value['populations'].pop()
    elif kind=='duplicate':value['populations'].append(value['populations'][0])
    elif kind=='overlap':value['populations'][0]['validation']=value['populations'][0]['training'][:1]
    else:value['populations'][0]['training'][0]['primary']=264
    with pytest.raises(ValueError):prepare(value)


def test_copy_and_deployment_boundary(monkeypatch):
    value=payload(3200);prepared=prepare(value)
    expected=prepared.payload['inference']['waveform'].copy()
    value['inference']['waveform'][:]=0
    np.testing.assert_array_equal(prepared.payload['inference']['waveform'],expected)
    monkeypatch.delenv('SCIONA_CORNELL_SOURCE_DIR',raising=False)
    with pytest.raises(ValueError,match='Provision'):execute(prepared)
    with pytest.raises(ValueError,match='Prepared'):execute({})

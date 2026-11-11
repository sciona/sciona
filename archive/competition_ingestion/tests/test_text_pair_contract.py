"""Strict input and complete synthetic boundary execution checks."""
import json
import pytest
from scripts.text_pair_synthetic import payload
from sciona.text_pair_contract import prepare,execute


def test_json_copy_and_complete_execution():
    p=payload();prepared=prepare(p);p['controls']['trees']=1
    result=execute(prepared)
    assert result['forest_trees']==64 and result['models']==2 and result['feature_count']==12
    assert [result[k] for k in ('training_rows','calibration_rows','query_rows')]==[8,4,2]
    assert json.loads(json.dumps(result,allow_nan=False))==result


@pytest.mark.parametrize('case',['extra','version','nan','tuple','controls','overlap','labels'])
def test_invalid_payload_rejected(case):
    p=payload()
    if case=='extra':p['extra']=1
    if case=='version':p['version']=True
    if case=='nan':p['controls']['seed']=float('nan')
    if case=='tuple':p['query_pairs'][0]=tuple(p['query_pairs'][0])
    if case=='controls':p['controls']['trees']=0
    if case=='overlap':p['query_pairs']=[p['calibration_pairs'][0][::-1]]
    if case=='labels':p['calibration_labels']=[1]*4
    with pytest.raises(ValueError):prepare(p)


def test_mutated_intermediate_rechecked_before_fit(monkeypatch):
    prepared=prepare(payload());prepared.payload['query_pairs']=prepared.payload['training_pairs']
    def forbidden(*args,**kwargs):raise AssertionError('Fit must not occur')
    monkeypatch.setattr('sciona.text_pair_contract.fit_ensemble',forbidden)
    with pytest.raises(ValueError):execute(prepared)

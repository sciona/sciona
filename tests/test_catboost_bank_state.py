import copy
import hashlib
import json
import pytest
import sciona.catboost_bank_state as state


def test_json_roundtrip_preserves_aliases_and_integer_slots(monkeypatch):
    first,second=object(),object()
    monkeypatch.setattr(state,'pack_catboost_model',lambda model: {'synthetic':0 if model is first else 1})
    monkeypatch.setattr(state,'validate_model_state',lambda payload: None)
    monkeypatch.setattr(state,'unpack_catboost_model',lambda payload: object())
    bank={'plant-A':{0:first,2:first,'shared':second},'plant-B':{'shared':second}}
    envelope=state.pack_model_bank(bank)
    restored=state.unpack_model_bank(json.loads(json.dumps(envelope)))
    assert restored['plant-A'][0] is restored['plant-A'][2]
    assert restored['plant-A']['shared'] is restored['plant-B']['shared']
    assert restored['plant-A'][0] is not restored['plant-A']['shared']
    assert len(envelope['payload']['objects'])==2
    bad=copy.deepcopy(envelope);bad['payload']['populations'][0]['bindings'][0]['object_index']=1
    with pytest.raises(ValueError,match='integrity'):state.unpack_model_bank(bad)
    bad['payload']['populations'][0]['bindings'][0]['object_index']=True
    bad['payload_sha256']=hashlib.sha256(state._canonical(bad['payload'])).hexdigest()
    with pytest.raises(ValueError,match='typed slot'):state.unpack_model_bank(bad)

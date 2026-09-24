import copy
import json
import numpy as np
import pandas as pd
import pytest
from sciona.catboost_model_state import pack_catboost_model,unpack_catboost_model,validate_model_state


@pytest.fixture(scope='module')
def native_state():
    from catboost import CatBoostRegressor
    frame=pd.DataFrame({'load':np.arange(32,dtype=float),'machine':['A','B']*16})
    model=CatBoostRegressor(iterations=5,depth=2,thread_count=1,allow_writing_files=False,verbose=False)
    model.fit(frame,np.arange(32,dtype=float)*2,cat_features=[1])
    return model,frame,pack_catboost_model(model)


def test_json_roundtrip_predictions_and_schema(native_state):
    model,frame,state=native_state
    restored=unpack_catboost_model(json.loads(json.dumps(state)))
    np.testing.assert_array_equal(model.predict(frame,thread_count=1),restored.predict(frame,thread_count=1))
    assert restored.feature_names_==model.feature_names_
    assert restored.get_cat_feature_indices()==[1]


@pytest.mark.parametrize('field,value',[
    ('backend_version','synthetic-invalid'),('payload','invalid!'),('payload_sha256','0'*64),
    ('feature_names',['x','x']),('categorical_indices',[2]),('categorical_indices',[True]),
])
def test_invalid_envelope_rejected(native_state,field,value):
    state=copy.deepcopy(native_state[2]);state[field]=value
    with pytest.raises(ValueError):validate_model_state(state)


def test_validly_encoded_schema_mismatch_rejected(native_state):
    state=copy.deepcopy(native_state[2]);state['feature_names']=['other','machine']
    with pytest.raises(ValueError,match='schema'):unpack_catboost_model(state)

import copy
import json

import numpy as np
import pytest

from sciona.atoms.ml.xgboost.model_io import fit_regression_model,fit_binary_model
from sciona.residual_model_state import pack_state,encode_state,decode_state,predict_state


@pytest.fixture(scope='module')
def state():
    rng=np.random.default_rng(3901)
    x=rng.normal(size=(32,2)).astype(np.float64)
    regressor=fit_regression_model(x,x[:,0]*2,['a','b'])
    classifier=fit_binary_model(np.column_stack([x,x[:,0]]),(x[:,0]>0).astype(np.int64),['a','b','prediction'])
    return pack_state(['a','b'],'prediction',regressor,classifier,np.array([1.,2.]),.5,{'domain':'synthetic_measurements'})


def test_complete_bundle_roundtrip_preserves_prediction_and_adapter_state(state):
    loaded=decode_state(encode_state(state))
    assert loaded==state
    query=np.array([[.5,1.],[-.7,.2]],dtype=np.float64)
    np.testing.assert_array_equal(predict_state(query,loaded),predict_state(query,state))
    loaded['adapter_state']['domain']='changed'
    assert state['adapter_state']['domain']=='synthetic_measurements'


def test_envelope_detects_modified_calibration_and_duplicate_keys(state):
    envelope=json.loads(encode_state(state))
    envelope['payload']['threshold']=.7
    with pytest.raises(ValueError,match='integrity'):
        decode_state(json.dumps(envelope))
    with pytest.raises(ValueError,match='Invalid encoded'):
        decode_state('{"payload":{},"payload":{},"payload_sha256":"x"}')


@pytest.mark.parametrize('fault',['feature_order','classifier_task','backend_version','payload',
    'offset_nan','threshold','feature_collision','adapter_nan','unknown_field'])
def test_incompatible_bundle_contracts_rejected(state,fault):
    value=copy.deepcopy(state)
    if fault=='feature_order':value['feature_names'].reverse()
    elif fault=='classifier_task':value['classifier']['task']='regression'
    elif fault=='backend_version':value['regressor']['backend_version']='incompatible'
    elif fault=='payload':value['regressor']['payload']='!'
    elif fault=='offset_nan':value['offsets'][0]=float('nan')
    elif fault=='threshold':value['threshold']=2.
    elif fault=='feature_collision':value['prediction_feature_name']='a'
    elif fault=='adapter_nan':value['adapter_state']['value']=float('nan')
    else:value['unexpected']=True
    with pytest.raises(ValueError):encode_state(value)

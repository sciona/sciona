"""Portable residual-classifier runtime state with explicit schema and calibration.

No file IO, pickle, dataset discovery or logging. State learned from non-public
inputs is private runtime material. Digests detect corruption, not authenticity.
"""
import base64
import copy
import hashlib
import importlib.metadata
import json
import math

import numpy as np

FORMAT = 'sciona.residual-classifier.state.v1'
FIELDS = {'format','feature_names','prediction_feature_name','regressor','classifier',
          'offsets','threshold','adapter_state'}


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _names(values):
    if not isinstance(values,list) or not values or any(not isinstance(value,str) or not value
            or any(character in value for character in '[]<') for value in values) or len(set(values))!=len(values):
        raise ValueError('Unique ordered backend-compatible feature schema required')


def validate_state(state):
    if not isinstance(state,dict) or set(state)!=FIELDS or state['format']!=FORMAT:
        raise ValueError('Complete residual-classifier state required')
    _names(state['feature_names'])
    _names(state['feature_names']+[state['prediction_feature_name']])
    expected={'regressor':('regression',state['feature_names']),
              'classifier':('binary',state['feature_names']+[state['prediction_feature_name']])}
    for key,(task,names) in expected.items():
        model=state[key]
        if not isinstance(model,dict) or set(model)!={'format','task','backend_version','feature_names','payload','payload_sha256'}:
            raise ValueError('Complete native model state required')
        if model['format']!='sciona.xgboost.model.v1' or model['task']!=task or model['feature_names']!=names or model['backend_version']!=importlib.metadata.version('xgboost'):
            raise ValueError('Native model task, backend or schema differs')
        try: payload=base64.b64decode(model['payload'],validate=True)
        except (ValueError,TypeError) as error: raise ValueError('Invalid native model encoding') from error
        if hashlib.sha256(payload).hexdigest()!=model['payload_sha256']:
            raise ValueError('Native model integrity differs')
    offsets=state['offsets']
    if not isinstance(offsets,list) or len(offsets)!=2 or any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in offsets):
        raise ValueError('Finite signed calibration offset pair required')
    threshold=state['threshold']
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Finite probability threshold required')
    if not isinstance(state['adapter_state'],dict):
        raise ValueError('Explicit JSON-compatible adapter state required')
    try: _canonical(state)
    except (TypeError,ValueError) as error: raise ValueError('JSON-compatible finite state required') from error


def pack_state(feature_names,prediction_feature_name,regressor,classifier,offsets,threshold,adapter_state):
    state=dict(format=FORMAT,feature_names=feature_names,prediction_feature_name=prediction_feature_name,
               regressor=regressor,classifier=classifier,offsets=np.asarray(offsets,dtype=np.float64).tolist(),
               threshold=threshold,adapter_state=adapter_state)
    validate_state(state)
    return copy.deepcopy(state)


def encode_state(state):
    validate_state(state)
    return _canonical(dict(payload=state,payload_sha256=hashlib.sha256(_canonical(state)).hexdigest()))


def decode_state(encoded):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate state key')
            result[key]=value
        return result
    def nonfinite(value):raise ValueError('Nonfinite JSON state rejected')
    try: envelope=json.loads(encoded,object_pairs_hook=pairs,parse_constant=nonfinite)
    except (TypeError,ValueError,UnicodeError) as error:raise ValueError('Invalid encoded state') from error
    if not isinstance(envelope,dict) or set(envelope)!={'payload','payload_sha256'}:
        raise ValueError('Complete state envelope required')
    if hashlib.sha256(_canonical(envelope['payload'])).hexdigest()!=envelope['payload_sha256']:
        raise ValueError('Residual state integrity differs')
    validate_state(envelope['payload'])
    return envelope['payload']


def predict_state(prediction_features,state):
    """Apply the published unrounded-feature, float32 correction, int32 path."""
    from sciona.atoms.ml.xgboost.model_io import predict_regression_model,predict_binary_model
    from sciona.atoms.ml.calibration.prediction_features import append_prediction_feature,round_to_int32
    from sciona.atoms.ml.calibration.float32_correction import apply_float32_offsets
    validate_state(state)
    predictions=predict_regression_model(prediction_features,state['feature_names'],state['regressor'])
    augmented=append_prediction_feature(prediction_features,predictions)
    probabilities=predict_binary_model(augmented,state['feature_names']+[state['prediction_feature_name']],state['classifier'])
    return round_to_int32(apply_float32_offsets(predictions,probabilities,np.array(state['offsets'],dtype=np.float64),state['threshold']))

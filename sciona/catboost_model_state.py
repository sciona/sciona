"""Explicit native regression state without pickle or embedded training records.

Learned payloads remain private runtime material for non-public inputs. Digests
detect corruption, not authenticity. Loading requires the recorded backend.
"""
import base64
import hashlib
import importlib.metadata
from pathlib import Path
import tempfile

FORMAT = 'sciona.catboost.regression-state.v1'
FIELDS = {'format','backend_version','feature_names','categorical_indices','payload','payload_sha256'}


def validate_model_state(state):
    if not isinstance(state,dict) or set(state)!=FIELDS or state['format']!=FORMAT:
        raise ValueError('Complete native regression state required')
    if state['backend_version']!=importlib.metadata.version('catboost'):
        raise ValueError('Exact native backend version required')
    names=state['feature_names'];categories=state['categorical_indices']
    if not isinstance(names,list) or not names or any(not isinstance(name,str) or not name for name in names) or len(set(names))!=len(names):
        raise ValueError('Unique ordered native feature names required')
    if not isinstance(categories,list) or any(type(index) is not int or not 0<=index<len(names) for index in categories) or len(set(categories))!=len(categories):
        raise ValueError('Unique valid native categorical indices required')
    if not isinstance(state['payload'],str):
        raise ValueError('Native model payload must be base64 text')
    try:
        payload=base64.b64decode(state['payload'],validate=True)
    except (ValueError,TypeError) as error:
        raise ValueError('Invalid native model encoding') from error
    if not payload or hashlib.sha256(payload).hexdigest()!=state['payload_sha256']:
        raise ValueError('Native model payload integrity differs')
    return payload


def pack_catboost_model(model):
    from catboost import CatBoostRegressor
    if not isinstance(model,CatBoostRegressor) or not model.is_fitted():
        raise ValueError('Fitted native regressor required')
    with tempfile.TemporaryDirectory(prefix='sciona-native-model-') as directory:
        path=Path(directory)/'model.cbm'
        model.save_model(str(path),format='cbm')
        payload=path.read_bytes()
    state=dict(format=FORMAT,backend_version=importlib.metadata.version('catboost'),
        feature_names=list(model.feature_names_),categorical_indices=list(model.get_cat_feature_indices()),
        payload=base64.b64encode(payload).decode('ascii'),payload_sha256=hashlib.sha256(payload).hexdigest())
    validate_model_state(state)
    return state


def unpack_catboost_model(state):
    from catboost import CatBoostRegressor
    payload=validate_model_state(state)
    with tempfile.TemporaryDirectory(prefix='sciona-native-model-') as directory:
        path=Path(directory)/'model.cbm';path.write_bytes(payload)
        model=CatBoostRegressor(thread_count=1)
        model.load_model(str(path),format='cbm')
    if list(model.feature_names_)!=state['feature_names'] or list(model.get_cat_feature_indices())!=state['categorical_indices']:
        raise ValueError('Restored native feature schema differs from envelope')
    return model

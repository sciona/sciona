"""Bind JSON-compatible model state and execution policy as one integrity envelope."""
import hashlib
import json


def _encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def bind_policy_state(model_state,policy):
    if not isinstance(model_state,dict) or not isinstance(policy,dict):
        raise ValueError('Explicit model state and policy mappings required')
    payload=dict(format='sciona.policy-bound-state.v1',model_state=model_state,policy=policy)
    encoded=_encode(payload)
    return dict(payload=json.loads(encoded),payload_sha256=hashlib.sha256(encoded).hexdigest())


def unbind_policy_state(state):
    if not isinstance(state,dict) or set(state)!={'payload','payload_sha256'}:
        raise ValueError('Complete policy-bound state required')
    payload=state['payload'];encoded=_encode(payload)
    if hashlib.sha256(encoded).hexdigest()!=state['payload_sha256']:
        raise ValueError('Model/policy binding integrity differs')
    if not isinstance(payload,dict) or set(payload)!={'format','model_state','policy'} or payload['format']!='sciona.policy-bound-state.v1':
        raise ValueError('Complete model/policy payload required')
    if not isinstance(payload['model_state'],dict) or not isinstance(payload['policy'],dict):
        raise ValueError('Model state and policy mappings required')
    owned=json.loads(encoded)
    return owned['model_state'],owned['policy']

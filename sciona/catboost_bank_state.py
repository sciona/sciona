"""Portable native model banks preserving shared objects and typed slot keys."""
import hashlib
import json
from sciona.catboost_model_state import pack_catboost_model,unpack_catboost_model,validate_model_state

FORMAT='sciona.catboost.model-bank.v1'


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def pack_model_bank(bank):
    if not isinstance(bank,dict) or not bank:
        raise ValueError('Nonempty named model bank required')
    objects=[];identities={};populations=[]
    for population,slots in bank.items():
        if not isinstance(population,str) or not population or not isinstance(slots,dict) or not slots:
            raise ValueError('Named populations with nonempty model slots required')
        bindings=[]
        for key,model in slots.items():
            if type(key) not in (int,str) or key=='':
                raise ValueError('Nonempty string or integer model slot required')
            identity=id(model)
            if identity not in identities:
                identities[identity]=len(objects);objects.append(pack_catboost_model(model))
            bindings.append(dict(slot=key,object_index=identities[identity]))
        populations.append(dict(population=population,bindings=bindings))
    payload=dict(format=FORMAT,objects=objects,populations=populations)
    return dict(payload=payload,payload_sha256=hashlib.sha256(_canonical(payload)).hexdigest())


def unpack_model_bank(envelope):
    if not isinstance(envelope,dict) or set(envelope)!={'payload','payload_sha256'}:
        raise ValueError('Complete model-bank envelope required')
    payload=envelope['payload']
    try:digest=hashlib.sha256(_canonical(payload)).hexdigest()
    except (TypeError,ValueError) as error:raise ValueError('Finite JSON model-bank payload required') from error
    if digest!=envelope['payload_sha256']:
        raise ValueError('Model-bank integrity differs')
    if not isinstance(payload,dict) or set(payload)!={'format','objects','populations'} or payload['format']!=FORMAT:
        raise ValueError('Complete typed model-bank payload required')
    states=payload['objects'];populations=payload['populations']
    if not isinstance(states,list) or not states or not isinstance(populations,list) or not populations:
        raise ValueError('Nonempty model objects and populations required')
    for state in states:validate_model_state(state)
    used=set();names=set()
    for item in populations:
        if not isinstance(item,dict) or set(item)!={'population','bindings'}:
            raise ValueError('Complete population bindings required')
        name=item['population'];bindings=item['bindings']
        if not isinstance(name,str) or not name or name in names or not isinstance(bindings,list) or not bindings:
            raise ValueError('Unique named populations with nonempty bindings required')
        names.add(name);slots=set()
        for binding in bindings:
            if not isinstance(binding,dict) or set(binding)!={'slot','object_index'}:
                raise ValueError('Complete typed model slot required')
            key,index=binding['slot'],binding['object_index']
            if type(key) not in (int,str) or key=='' or key in slots or type(index) is not int or not 0<=index<len(states):
                raise ValueError('Unique typed slot and valid model index required')
            slots.add(key);used.add(index)
    if used!=set(range(len(states))):raise ValueError('Unreferenced native model objects')
    # Validate all references before loading any native payload.
    models=[unpack_catboost_model(state) for state in states]
    return {item['population']:{binding['slot']:models[binding['object_index']] for binding in item['bindings']}
            for item in populations}

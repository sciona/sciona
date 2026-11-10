"""Numeric-only B5 prediction state; no executable graph or optimizer restore."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from sciona.osic_image_model import build_model

FORMAT='sciona.osic_image_prediction.v1'
ARCHITECTURE='osic.random_initialized_b5_slope.v1'
FIELDS={'format','architecture','image_coordinate','metadata_coordinate','response_coordinate',
        'weights_sha256','weight_shapes','model_code_sha256','graph_sha256'}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _coordinates(image_coordinate,metadata_coordinate,response_coordinate):
    values=(image_coordinate,metadata_coordinate,response_coordinate)
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError('Explicit nonempty image, metadata and response coordinates required')
    return dict(zip(('image_coordinate','metadata_coordinate','response_coordinate'),values))


def _graph_hash(model):
    def normalize(config):
        config=copy.deepcopy(config)
        mapping={layer['name']:'layer_'+str(i) for i,layer in enumerate(config['layers'])}
        def references(value):
            if isinstance(value,(list,tuple)):
                result=[references(item) for item in value]
                if result and isinstance(result[0],str) and result[0] in mapping:
                    result[0]=mapping[result[0]]
                return result
            if isinstance(value,dict):
                return {key:references(item) for key,item in value.items()}
            return value
        config['name']='model'
        for layer in config['layers']:
            name=mapping[layer['name']]
            layer['name']=name
            if 'layers' in layer['config']:
                layer['config']=normalize(layer['config'])
            else:
                layer['config']['name']=name
            layer['inbound_nodes']=references(layer['inbound_nodes'])
        for key in ('input_layers','output_layers'):
            config[key]=references(config[key])
        return config
    raw=json.dumps(normalize(model.get_config()),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def export_prediction_state(model,directory,*,image_coordinate,metadata_coordinate,response_coordinate):
    """Write private NPZ arrays and a strict JSON manifest for inference only.

An observed positive optimizer counter is required, but is not proof of the
complete training schedule. Graph topology is compared with the local fixed
constructor after normalizing only layer names and their corresponding edges.
The constructor consumes initialization randomness during validation. Call this
outside live training. Artifact authenticity remains the caller's responsibility.
"""
    coordinates=_coordinates(image_coordinate,metadata_coordinate,response_coordinate)
    reference=build_model()
    graph_hash=_graph_hash(reference)
    optimizer=getattr(model,'optimizer',None)
    if (type(model) is not type(reference) or _graph_hash(model)!=graph_hash
            or optimizer is None or int(optimizer.iterations.numpy())<1):
        raise ValueError('An updated instance of the exact known B5 architecture is required')
    shapes=[tuple(int(d) for d in weight.shape) for weight in reference.weights]
    weights=model.get_weights()
    if len(weights)!=len(shapes) or any(value.shape!=shape or value.dtype!=np.float32
            or not np.isfinite(value).all() for value,shape in zip(weights,shapes)):
        raise ValueError('Complete finite float32 weights required')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    np.savez(directory/'weights.npz',**{'tensor_'+str(i):value for i,value in enumerate(weights)})
    manifest=dict(format=FORMAT,architecture=ARCHITECTURE,**coordinates,
        weights_sha256=_sha(directory/'weights.npz'),weight_shapes=[list(shape) for shape in shapes],
        model_code_sha256=_sha(Path(__file__).with_name('osic_image_model.py')),graph_sha256=graph_hash)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def restore_prediction_state(directory,*,image_coordinate,metadata_coordinate,response_coordinate):
    """Reconstruct locally, validate numeric arrays, and load weights without fitting."""
    coordinates=_coordinates(image_coordinate,metadata_coordinate,response_coordinate)
    directory=Path(directory)
    state=json.loads((directory/'manifest.json').read_text())
    if (type(state) is not dict or set(state)!=FIELDS or state['format']!=FORMAT
            or state['architecture']!=ARCHITECTURE
            or any(state[key]!=value for key,value in coordinates.items())
            or state['model_code_sha256']!=_sha(Path(__file__).with_name('osic_image_model.py'))):
        raise ValueError('Exact format, local architecture and matching coordinates required')
    if state['weights_sha256']!=_sha(directory/'weights.npz'):
        raise ValueError('Numeric weight archive changed')
    model=build_model()
    shapes=[tuple(int(d) for d in weight.shape) for weight in model.weights]
    if state['graph_sha256']!=_graph_hash(model) or state['weight_shapes']!=[list(shape) for shape in shapes]:
        raise ValueError('Local graph or weight inventory differs')
    with np.load(directory/'weights.npz',allow_pickle=False) as archive:
        names=['tensor_'+str(i) for i in range(len(shapes))]
        if archive.files!=names:
            raise ValueError('Exact ordered numeric weight inventory required')
        weights=[archive[name] for name in names]
    if any(value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all()
           for value,shape in zip(weights,shapes)):
        raise ValueError('Exact finite float32 weight shapes required')
    model.set_weights(weights)
    return model

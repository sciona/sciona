"""Validate the numerical residual graph after upstream arrays materialize.

Metadata propagation uses observed array shapes, never guessed join cardinality.
This does not prove the later model-dependent selected populations or quality.
"""
import math

import numpy as np

from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import REGISTRY
from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.visualizer.runner import _ensure_atoms_imported, get_topo_sorted_leaves

KEYS = ('features','targets','groups','feature_names','prediction_features',
        'train_fraction','seed','maximum_error','threshold','prediction_feature_name')


def propagate_metadata(roots):
    _ensure_atoms_imported()
    graph = build_residual_classifier_graph()
    state = {}
    for node in get_topo_sorted_leaves(graph.nodes, graph.edges):
        arguments = {}
        for port in node.inputs:
            sources = [edge for edge in graph.edges if edge.target_id==node.node_id and edge.input_name==port.name]
            if len(sources)>1:
                raise ValueError('Ambiguous numerical metadata input')
            arguments[port.name] = state[sources[0].source_id,sources[0].output_name] if sources else roots[port.name]
        result = REGISTRY[node.matched_primitive]['witness'](**arguments)
        values = result if len(node.outputs)>1 else (result,)
        if not isinstance(values, tuple) or len(values)!=len(node.outputs):
            raise ValueError('Numerical metadata output arity differs')
        state.update({(node.node_id,port.name):value for port,value in zip(node.outputs,values)})
    final = state['final','predictions']
    if final.shape != (roots['prediction_features'].shape[0],) or final.dtype!='int32' or final.dim!=roots['targets'].dim:
        raise ValueError('Numerical metadata output contract differs')
    return dict(nodes_visited=len(graph.nodes), final=final)


def validate_materialized_inputs(payload, target_dimension):
    if set(payload)!=set(KEYS):
        raise ValueError('Complete explicit numerical inputs required')
    for key, dtype, rank in [('features',np.float64,2),('targets',np.float64,1),
                             ('groups',np.int64,1),('prediction_features',np.float64,2)]:
        value = payload[key]
        if not isinstance(value,np.ndarray) or value.dtype!=dtype or value.ndim!=rank or min(value.shape)<1:
            raise ValueError('Materialized array shape/dtype differs: '+key)
        if np.isinf(value).any() or key=='targets' and not np.isfinite(value).all():
            raise ValueError('Invalid materialized numerical values: '+key)
    if len(np.unique(payload['groups']))<2:
        raise ValueError('At least two training groups required')
    for name in ['train_fraction','maximum_error','threshold']:
        value = payload[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Finite numerical control required: '+name)
    if not 0<payload['train_fraction']<1 or payload['maximum_error']<0 or not 0<=payload['threshold']<=1:
        raise ValueError('Numerical control outside supported range')
    if type(payload['seed']) is not int or not 0<=payload['seed']<=2**32-1:
        raise ValueError('Unsigned 32-bit split seed required')
    roots = {key:AbstractArray(shape=payload[key].shape,dtype=str(payload[key].dtype),
                              dim=target_dimension if key=='targets' else None)
             for key in ['features','targets','groups','prediction_features']}
    roots.update(feature_names=payload['feature_names'],prediction_feature_name=payload['prediction_feature_name'],
        train_fraction=AbstractScalar(dtype='float64'),seed=AbstractScalar(dtype='int64'),
        maximum_error=AbstractScalar(dtype='float64',dim=target_dimension),threshold=AbstractScalar(dtype='float64'))
    return propagate_metadata(roots)

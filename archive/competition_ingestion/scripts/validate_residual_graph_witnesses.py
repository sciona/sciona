"""Propagate registered witnesses over every edge of the numerical CDG."""
import hashlib
import json
from pathlib import Path

from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import REGISTRY
from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.visualizer.runner import _ensure_atoms_imported, get_topo_sorted_leaves

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return dict(features=AbstractArray(shape=(480, 3)),
        targets=AbstractArray(shape=(480,), dim=DimensionalSignature(L=1)),
        groups=AbstractArray(shape=(480,), dtype='int64'), feature_names=['a', 'b', 'c'],
        prediction_features=AbstractArray(shape=(64, 3)),
        train_fraction=AbstractScalar(dtype='float64'), seed=AbstractScalar(dtype='int64'),
        maximum_error=AbstractScalar(dtype='float64', dim=DimensionalSignature(L=1)),
        threshold=AbstractScalar(dtype='float64'), prediction_feature_name='predicted')


def propagate(graph, roots, overrides=None):
    _ensure_atoms_imported()
    state = {}
    visited = []
    for node in get_topo_sorted_leaves(graph.nodes, graph.edges):
        arguments = {}
        for port in node.inputs:
            sources = [edge for edge in graph.edges if edge.target_id == node.node_id and edge.input_name == port.name]
            if len(sources) > 1:
                raise ValueError('Ambiguous symbolic input binding')
            if sources:
                edge = sources[0]
                arguments[port.name] = state[(edge.source_id, edge.output_name)]
            else:
                arguments[port.name] = roots[port.name]
        result = REGISTRY[node.matched_primitive]['witness'](**arguments)
        values = result if len(node.outputs) > 1 else (result,)
        if len(values) != len(node.outputs):
            raise ValueError('Symbolic output arity differs')
        for port, value in zip(node.outputs, values):
            key = node.node_id, port.name
            state[key] = (overrides or {}).get(key, value)
        visited.append(node.node_id)
    if len(visited) != len(graph.nodes):
        raise ValueError('Incomplete graph traversal')
    return state, visited


def validate():
    graph = build_residual_classifier_graph()
    state, visited = propagate(graph, inputs())
    final = state[('final', 'predictions')]
    if final.shape != (64,) or final.dtype != 'int32' or final.dim != DimensionalSignature(L=1):
        raise ValueError('Final target dimension, shape or dtype lost')
    faults = []
    for name in ['target_rows', 'query_columns', 'threshold_units', 'held_mask_rows', 'classifier_task']:
        roots, overrides = inputs(), {}
        if name == 'target_rows': roots['targets'] = AbstractArray(shape=(479,))
        elif name == 'query_columns': roots['prediction_features'] = AbstractArray(shape=(64, 4))
        elif name == 'threshold_units': roots['maximum_error'] = AbstractScalar(dtype='float64', dim=DimensionalSignature(T=1))
        elif name == 'held_mask_rows': overrides[('split', 'held')] = AbstractArray(shape=(479,), dtype='bool')
        else:
            bad = dict(state[('classifier_fit', 'state')]); bad['task'] = 'regression'
            overrides[('classifier_fit', 'state')] = bad
        try:
            propagate(graph, roots, overrides)
        except ValueError:
            faults.append(name)
        else:
            raise ValueError('Symbolic fault escaped: '+name)
    return dict(passed=True, approved=False, catalog_mutations=0, graph_sha256=encode_execution_graph(graph)[0],
        nodes_visited=len(visited), edges=len(graph.edges), final_shape=list(final.shape), final_dtype=final.dtype,
        target_dimension_preserved=True, rejected_faults=faults,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Direct topological execution of registered witnesses; no model fitting or runtime values used.',
            'Heterogeneous per-column feature units remain an external schema contract.',
            'Witnesses cannot prove sufficient selected rows, both classes, finite residuals or populated calibration groups; runtime checks remain required.',
            'Domain adapter graph and publication qualification remain pending.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/residual_classifier_graph_witnesses.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))

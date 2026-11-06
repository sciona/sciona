"""Draft tracking graph with explicit reusable fitted neighbor operations."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.ml.sklearn.neighbors.fitted_search import (
    fit_euclidean_search, query_euclidean_k, query_euclidean_radius,
)
from sciona.atoms.physics.decomposed_trackml_neighbor_execution import track
from sciona.decomposed_trackml_correction_graph import build_decomposed_trackml_correction_graph


def build_decomposed_trackml_neighbor_graph():
    graph = build_decomposed_trackml_correction_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    for name, function, outputs in [
        ('search_fit', fit_euclidean_search, [('state', 'object')]),
        ('search_k', query_euclidean_k, [('distances', 'ndarray'), ('indices', 'ndarray')]),
        ('search_radius', query_euclidean_radius, [('distances', 'ndarray'), ('indices', 'ndarray')]),
    ]:
        input_name = name + '_function'
        tracking.inputs.append(IOSpec(name=input_name, type_desc='object'))
        types = dict(points='ndarray', state='object', default_neighbors='int', leaf_size='int', k='int', radius='float')
        inputs = [IOSpec(name=p.name, type_desc=types[p.name], required=p.default is p.empty,
                         default_value_repr='' if p.default is p.empty else repr(p.default))
                  for p in inspect.signature(function).parameters.values()]
        graph.nodes.append(AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(function),
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__ + '.' + function.__name__, inputs=inputs,
            outputs=[IOSpec(name=output, type_desc=kind) for output, kind in outputs]))
        graph.edges.append(DependencyEdge(source_id=name, target_id='track', output_name=outputs[0][0],
            input_name=input_name, source_type='object', target_type='object', edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_neighbor_decomposition', num_nodes=16, num_edges=15,
        scope='Every configured tracking round with thirteen separately bound reusable numeric providers.')
    return graph

"""Draft tracking graph with explicit cylinder and plane projection providers."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.physics.coordinate_projection import cylindrical_projection, plane_projection
from sciona.atoms.physics.decomposed_trackml_coordinate_execution import track
from sciona.decomposed_trackml_projection_graph import build_decomposed_trackml_projection_graph


def build_decomposed_trackml_coordinate_graph():
    graph = build_decomposed_trackml_projection_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    for name, function in [('cylinder', cylindrical_projection), ('plane', plane_projection)]:
        input_name = name + '_function'
        tracking.inputs.append(IOSpec(name=input_name, type_desc='object'))
        inputs = [IOSpec(name=p.name, type_desc='ndarray' if p.name == 'coords' else 'float',
                         required=p.default is p.empty,
                         default_value_repr='' if p.default is p.empty else repr(p.default))
                  for p in inspect.signature(function).parameters.values()]
        graph.nodes.append(AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(function),
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__ + '.' + function.__name__, inputs=inputs,
            outputs=[IOSpec(name='projected', type_desc='ndarray')]))
        graph.edges.append(DependencyEdge(source_id=name, target_id='track', output_name='projected',
            input_name=input_name, source_type='object', target_type='object', edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_coordinate_decomposition', num_nodes=9, num_edges=8,
        scope='Every configured tracking round with six separately bound reusable numeric providers.')
    return graph

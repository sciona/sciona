"""Draft tracking graph with explicit reusable cylinder and plane intersections."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.physics.helix_cylinder import helix_cylinder_intersection
from sciona.atoms.particle_tracking.track_matching.atoms import helix_cap_intersection
from sciona.atoms.physics.decomposed_trackml_intersection_execution import track
from sciona.decomposed_trackml_coordinate_graph import build_decomposed_trackml_coordinate_graph


def build_decomposed_trackml_intersection_graph():
    graph = build_decomposed_trackml_coordinate_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    for name, function, outputs in [
        ('cylinder_intersection', helix_cylinder_intersection, ('x', 'y', 'z', 'phase', 'hit')),
        ('plane_intersection', helix_cap_intersection, ('x', 'y', 'z', 'phase')),
    ]:
        input_name = name + '_function'
        tracking.inputs.append(IOSpec(name=input_name, type_desc='object'))
        inputs = [IOSpec(name=p.name, type_desc='ndarray', required=p.default is p.empty,
                         default_value_repr='' if p.default is p.empty else repr(p.default))
                  for p in inspect.signature(function).parameters.values()]
        graph.nodes.append(AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(function),
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__ + '.' + function.__name__, inputs=inputs,
            outputs=[IOSpec(name=output, type_desc='ndarray') for output in outputs]))
        graph.edges.append(DependencyEdge(source_id=name, target_id='track', output_name=outputs[0],
            input_name=input_name, source_type='object', target_type='object', edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_intersection_decomposition', num_nodes=11, num_edges=10,
        scope='Every configured tracking round with eight separately bound reusable numeric providers.')
    return graph

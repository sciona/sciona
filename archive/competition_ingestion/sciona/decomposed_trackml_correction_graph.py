"""Draft tracking graph with explicit reusable cap and cylinder corrections."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.particle_tracking.detector_corrections.atoms import perturbative_cap_correction, perturbative_cylinder_correction
from sciona.atoms.physics.decomposed_trackml_correction_execution import track
from sciona.decomposed_trackml_intersection_graph import build_decomposed_trackml_intersection_graph


def build_decomposed_trackml_correction_graph():
    graph = build_decomposed_trackml_intersection_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    for name, function, outputs in [
        ('cap_correction', perturbative_cap_correction, ('x', 'y')),
        ('cylinder_correction', perturbative_cylinder_correction, ('x', 'y', 'z')),
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
    graph.metadata.update(artifact_source='tracking_correction_decomposition', num_nodes=13, num_edges=12,
        scope='Every configured tracking round with ten separately bound reusable numeric providers.')
    return graph

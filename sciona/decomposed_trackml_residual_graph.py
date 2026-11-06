"""Draft tracking graph with explicit reusable residual arithmetic."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.physics.tangent_residuals import hemisphere_tangent_displacement, combine_scaled_uncertainty
from sciona.atoms.physics.decomposed_trackml_residual_execution import track
from sciona.decomposed_trackml_detector_graph import build_decomposed_trackml_detector_graph


def build_decomposed_trackml_residual_graph():
    graph = build_decomposed_trackml_detector_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    for name, function, outputs in [
        ('displacement', hemisphere_tangent_displacement, ('polar', 'azimuth')),
        ('uncertainty', combine_scaled_uncertainty, ('result',)),
    ]:
        port = name + '_function'
        tracking.inputs.append(IOSpec(name=port, type_desc='object'))
        inputs = [IOSpec(name=p.name, type_desc='float' if p.name.endswith('_scale') else 'ndarray')
                  for p in inspect.signature(function).parameters.values()]
        graph.nodes.append(AlgorithmicNode(node_id=name, name=name+'_arithmetic',
            description=inspect.getdoc(function), concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__+'.'+function.__name__, inputs=inputs,
            outputs=[IOSpec(name=output, type_desc='ndarray') for output in outputs]))
        graph.edges.append(DependencyEdge(source_id=name, target_id='track', output_name=outputs[0],
            input_name=port, source_type='object', target_type='object', edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_residual_decomposition', num_nodes=22, num_edges=21,
        scope='Configured tracking rounds with nineteen explicit reusable numeric providers.')
    return graph

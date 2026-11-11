"""Draft tracking graph with explicit reusable tangent geometry callables."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.physics.helix_tangent import construct_helix_from_tangent, oriented_helix_tangent
from sciona.atoms.physics.decomposed_trackml_tangent_execution import track
from sciona.decomposed_trackml_direction_graph import build_decomposed_trackml_direction_graph


def build_decomposed_trackml_tangent_graph():
    graph = build_decomposed_trackml_direction_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__+'.'+track.__name__
    tracking.description = inspect.getdoc(track)
    for name,function,outputs in (
            ('construction',construct_helix_from_tangent,['centers','radius']),
            ('tangent',oriented_helix_tangent,['tangent'])):
        port = name+'_function'
        tracking.inputs.append(IOSpec(name=port,type_desc='object'))
        graph.nodes.append(AlgorithmicNode(node_id=name,name=name,
            description=inspect.getdoc(function),concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__+'.'+function.__name__,
            inputs=[IOSpec(name=p.name,type_desc='float' if p.annotation is float else 'ndarray')
                    for p in inspect.signature(function).parameters.values()],
            outputs=[IOSpec(name=p,type_desc='ndarray') for p in outputs]))
        graph.edges.append(DependencyEdge(source_id=name,target_id='track',output_name=outputs[0],
            input_name=port,source_type='object',target_type='object',edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_tangent_decomposition',num_nodes=27,num_edges=26,
        scope='Configured tracking rounds with twenty-four explicit reusable numeric providers.')
    return graph

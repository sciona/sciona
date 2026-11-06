"""Draft tracking graph with an explicit reusable helix-direction callable."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.particle_tracking.helix_geometry.atoms import helix_direction_from_two_points
from sciona.atoms.physics.decomposed_trackml_direction_execution import track
from sciona.decomposed_trackml_grid_graph import build_decomposed_trackml_grid_graph


def build_decomposed_trackml_direction_graph():
    graph = build_decomposed_trackml_grid_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__+'.'+track.__name__
    tracking.description = inspect.getdoc(track)
    tracking.inputs.append(IOSpec(name='direction_function',type_desc='object'))
    function = helix_direction_from_two_points
    graph.nodes.append(AlgorithmicNode(node_id='direction',name='direction',
        description=inspect.getdoc(function),concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive=function.__module__+'.'+function.__name__,
        inputs=[IOSpec(name=p.name,type_desc='ndarray') for p in inspect.signature(function).parameters.values()],
        outputs=[IOSpec(name='hel_dz',type_desc='ndarray')]))
    graph.edges.append(DependencyEdge(source_id='direction',target_id='track',output_name='hel_dz',
        input_name='direction_function',source_type='object',target_type='object',edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_direction_decomposition',num_nodes=25,num_edges=24,
        scope='Configured tracking rounds with twenty-two explicit reusable numeric providers.')
    return graph

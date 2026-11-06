"""Draft tracking graph with an explicit reusable nearest-point callable edge."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.particle_tracking.helix_geometry.atoms import helix_nearest_point_distance
from sciona.atoms.physics.decomposed_trackml_projection_execution import track
from sciona.decomposed_trackml_graph import build_decomposed_trackml_graph


def build_decomposed_trackml_projection_graph():
    graph = build_decomposed_trackml_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    tracking.inputs.append(IOSpec(name='nearest_function', type_desc='object'))
    function = helix_nearest_point_distance
    inputs = [IOSpec(name=p.name, type_desc='int' if p.name == 'iterations' else 'ndarray',
                     required=p.default is p.empty,
                     default_value_repr='' if p.default is p.empty else repr(p.default))
              for p in inspect.signature(function).parameters.values()]
    graph.nodes.append(AlgorithmicNode(node_id='nearest', name='nearest',
        description='Fixed-iteration nearest-helix approximation; no global-optimum guarantee.',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive=function.__module__ + '.' + function.__name__, inputs=inputs,
        outputs=[IOSpec(name=name, type_desc='ndarray') for name in ('x1', 'y1', 'z1', 'dist')]))
    graph.edges.append(DependencyEdge(source_id='nearest', target_id='track', output_name='x1',
        input_name='nearest_function', source_type='object', target_type='object',
        edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_projection_decomposition', num_nodes=7, num_edges=6,
        scope='Every configured tracking round with four separately bound reusable geometry providers.')
    return graph

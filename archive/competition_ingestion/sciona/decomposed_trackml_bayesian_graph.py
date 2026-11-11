"""Draft tracking graph with an explicit reusable normalized decision provider."""
import inspect

from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.particle_tracking.track_matching.atoms import bayesian_neighbor_evaluation
from sciona.atoms.physics.decomposed_trackml_bayesian_execution import track
from sciona.decomposed_trackml_neighbor_graph import build_decomposed_trackml_neighbor_graph


def build_decomposed_trackml_bayesian_graph():
    graph = build_decomposed_trackml_neighbor_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    tracking.inputs.append(IOSpec(name='decision_function', type_desc='object'))
    function = bayesian_neighbor_evaluation
    inputs = [IOSpec(name=p.name, type_desc='float' if p.name in ('cut_factor', 'dist_trust') else 'ndarray',
                     required=p.default is p.empty,
                     default_value_repr='' if p.default is p.empty else repr(p.default))
              for p in inspect.signature(function).parameters.values()]
    graph.nodes.append(AlgorithmicNode(node_id='decision', name='normalized_residual_decision',
        description=inspect.getdoc(function), concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive=function.__module__ + '.' + function.__name__, inputs=inputs,
        outputs=[IOSpec(name='good_neighbor', type_desc='ndarray'), IOSpec(name='dubious', type_desc='ndarray')]))
    graph.edges.append(DependencyEdge(source_id='decision', target_id='track', output_name='good_neighbor',
        input_name='decision_function', source_type='object', target_type='object',
        edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_bayesian_decomposition', num_nodes=17, num_edges=16,
        scope='Every configured tracking round with fourteen separately bound reusable numeric providers.')
    return graph

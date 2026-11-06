"""Draft tracking graph with explicit reusable ordered resource selection."""
import inspect

from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.particle_tracking.track_matching.atoms import greedy_track_commit
from sciona.atoms.physics.decomposed_trackml_commit_execution import track
from sciona.decomposed_trackml_bayesian_graph import build_decomposed_trackml_bayesian_graph


def build_decomposed_trackml_commit_graph():
    graph = build_decomposed_trackml_bayesian_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    tracking.inputs.append(IOSpec(name='commit_function', type_desc='object'))
    function = greedy_track_commit
    types = dict(hit_matrix='ndarray', used='ndarray', min_nhits='int', max_nloss='int',
                 max_loss_fraction='float', max_nrows='int')
    inputs = [IOSpec(name=p.name, type_desc=types[p.name], required=p.default is p.empty,
                     default_value_repr='' if p.default is p.empty else repr(p.default))
              for p in inspect.signature(function).parameters.values()]
    graph.nodes.append(AlgorithmicNode(node_id='commit', name='ordered_resource_selection',
        description=inspect.getdoc(function), concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive=function.__module__ + '.' + function.__name__, inputs=inputs,
        outputs=[IOSpec(name='selected', type_desc='ndarray'), IOSpec(name='claimed', type_desc='ndarray')]))
    graph.edges.append(DependencyEdge(source_id='commit', target_id='track', output_name='selected',
        input_name='commit_function', source_type='object', target_type='object',
        edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_commit_decomposition', num_nodes=18, num_edges=17,
        scope='Every configured tracking round with fifteen separately bound reusable numeric providers.')
    return graph

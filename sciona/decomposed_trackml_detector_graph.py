"""Draft tracking graph with explicit generic clustering providers."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.ml.sklearn.cluster.lloyd_fit import fit_lloyd_clusters
from sciona.atoms.ml.sklearn.cluster.atoms import mean_shift
from sciona.atoms.physics.decomposed_trackml_detector_execution import track
from sciona.decomposed_trackml_commit_graph import build_decomposed_trackml_commit_graph


def build_decomposed_trackml_detector_graph():
    graph = build_decomposed_trackml_commit_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    for name, function in [('kmeans', fit_lloyd_clusters), ('mean_shift', mean_shift)]:
        port = name + '_function'
        tracking.inputs.append(IOSpec(name=port, type_desc='object'))
        types = dict(points='ndarray', X='ndarray', n_clusters='int', random_state='int',
                     n_init='int', max_iter='int', tolerance='float', bandwidth='float',
                     seeds='ndarray', bin_seeding='bool', min_bin_freq='int',
                     cluster_all='bool', n_jobs='int')
        inputs = [IOSpec(name=p.name, type_desc=types[p.name], required=p.default is p.empty,
                         default_value_repr='' if p.default is p.empty else repr(p.default))
                  for p in inspect.signature(function).parameters.values()]
        graph.nodes.append(AlgorithmicNode(node_id=name, name=name+'_clustering',
            description=inspect.getdoc(function), concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__+'.'+function.__name__, inputs=inputs,
            outputs=[IOSpec(name='centers', type_desc='ndarray'), IOSpec(name='labels', type_desc='ndarray')]))
        graph.edges.append(DependencyEdge(source_id=name, target_id='track', output_name='centers',
            input_name=port, source_type='object', target_type='object', edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_detector_decomposition', num_nodes=20, num_edges=19,
        scope='Configured tracking rounds with seventeen explicit reusable numeric providers including clustering.')
    return graph

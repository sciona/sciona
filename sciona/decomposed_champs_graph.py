"""Complete CHAMPS computation with an explicit domain-independent rank mean."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.champs_graph import build_champs_graph


def build_decomposed_champs_graph():
    graph = build_champs_graph()
    prepare, training = graph.nodes
    prefix = 'sciona.atoms.ml.decomposed_champs_execution.'
    training.node_id = training.name = 'train_models'
    training.matched_primitive = prefix + 'train_models'
    training.description = 'Complete thirteen-model training and prediction before ensemble reduction'
    training.outputs = [IOSpec(name='training_result', type_desc='dict')]
    def node(name, description, fqdn, inputs, outputs):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=fqdn,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs])
    graph.nodes = [prepare, training,
        node('ensemble_inputs', 'Align identities and apply source coupling-specific model selection',
             prefix + 'ensemble_inputs', [('training_result', 'dict')],
             [('keys', 'tuple'), ('values', 'numpy.ndarray'), ('retained', 'int')]),
        node('central_mean', 'Average a caller-sized central rank interval independently per column',
             'sciona.atoms.numerical.statistics.central_mean.central_rank_mean',
             [('values', 'numpy.ndarray'), ('retained', 'int')], [('values', 'numpy.ndarray')]),
        node('result', 'Restore source result and complete training evidence', prefix + 'result',
             [('training_result', 'dict'), ('keys', 'tuple'), ('values', 'numpy.ndarray')], [('result', 'dict')])]
    links = [('prepare', 'train_models', 'prepared', 'object'),
             ('train_models', 'ensemble_inputs', 'training_result', 'dict'),
             ('ensemble_inputs', 'central_mean', 'values', 'numpy.ndarray'),
             ('ensemble_inputs', 'central_mean', 'retained', 'int'),
             ('train_models', 'result', 'training_result', 'dict'),
             ('ensemble_inputs', 'result', 'keys', 'tuple'),
             ('central_mean', 'result', 'values', 'numpy.ndarray')]
    graph.edges = [DependencyEdge(source_id=s, target_id=t, output_name=p, input_name=p,
                                 source_type=k, target_type=k) for s, t, p, k in links]
    graph.metadata.update(num_nodes=5, num_edges=7,
        catalog_output_sources={'result': dict(node_id='result', output_name='result')},
        reuse_boundary='Numeric central-rank mean is generic; chemistry, identities, model masks, retained count and training lifecycle remain recipe policy.',
        qualification_status='Full graph execution, current provider eligibility and original intake publication remain pending.')
    return graph

"""Complete APTOS reference with explicit reusable cut-point indexing."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from scripts.build_aptos_execution_graph import build_graph


def build_decomposed_aptos_graph():
    graph = build_graph()
    population, training = graph.nodes
    training.node_id = training.name = 'train_scores'
    training.description = 'Train all eight models and both complete stages, returning scores before class indexing'
    prefix = 'sciona.atoms.ml.decomposed_aptos_execution.'
    training.matched_primitive = prefix+'train_scores'
    training.outputs = [IOSpec(name='training_result', type_desc='dict',
        constraints='Complete training result before ordinal indexing; includes source tie policy and all fit/replay evidence')]
    def node(key, description, fqdn, inputs, outputs):
        return AlgorithmicNode(node_id=key, name=key, description=description, concept_type='custom',
            status=NodeStatus.ATOMIC, matched_primitive=fqdn,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs])
    graph.nodes = [population, training,
        node('cut_inputs', 'Expose source cut points and explicit equality policy', prefix+'cut_inputs',
            [('training_result', 'dict')], [('values', 'numpy.ndarray'), ('cut_points', 'numpy.ndarray'), ('tie_policy', 'str')]),
        node('classify', 'Index scores between caller-supplied ordered boundaries',
            'sciona.atoms.numerical.statistics.cut_indices.ordered_cut_indices',
            [('values', 'numpy.ndarray'), ('cut_points', 'numpy.ndarray'), ('tie_policy', 'str')], [('ordinals', 'numpy.ndarray')]),
        node('result', 'Attach class indices to the complete training result', prefix+'result',
            [('training_result', 'dict'), ('ordinals', 'numpy.ndarray')], [('result', 'dict')])]
    links = [('population', 'train_scores', 'plan', 'object'),
        ('train_scores', 'cut_inputs', 'training_result', 'dict'),
        ('cut_inputs', 'classify', 'values', 'numpy.ndarray'),
        ('cut_inputs', 'classify', 'cut_points', 'numpy.ndarray'),
        ('cut_inputs', 'classify', 'tie_policy', 'str'),
        ('train_scores', 'result', 'training_result', 'dict'),
        ('classify', 'result', 'ordinals', 'numpy.ndarray')]
    graph.edges = [DependencyEdge(source_id=s, target_id=t, output_name=p, input_name=p,
        source_type=k, target_type=k) for s, t, p, k in links]
    graph.metadata.update(num_nodes=5, num_edges=7,
        catalog_output_sources={'result': dict(node_id='result', output_name='result')},
        reuse_boundary='Cut-point indexing is domain independent. Four fixed source cuts, model families, population roles and training controls remain recipe policy.',
        qualification_status='Full decomposed execution and provider publication remain pending.')
    return graph

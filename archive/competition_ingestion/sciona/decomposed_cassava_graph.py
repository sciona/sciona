"""Full Cassava recipe with two explicitly bound domain-independent sums."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from scripts.build_cassava_execution_graph import build_graph


def build_decomposed_cassava_graph():
    graph = build_graph()
    plan, training = graph.nodes
    prefix = 'sciona.atoms.ml.decomposed_cassava_execution.'
    generic = 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'
    training.node_id = training.name = 'train_families'
    training.matched_primitive = prefix+'train_families'
    training.description = 'Train all three fivefold families, refit B4 and run frozen CropNet before final assembly'
    training.outputs = [IOSpec(name='training_result', type_desc='dict',
        constraints='Complete same-invocation family outputs, query keys, histories, evaluations and checkpoints')]
    def node(key, description, fqdn, inputs, outputs):
        return AlgorithmicNode(node_id=key, name=key, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=fqdn,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs])
    graph.nodes = [plan, training,
        node('pair_inputs', 'Align source family probabilities and supply unscaled pair operands', prefix+'pair_inputs',
            [('training_result', 'dict')], [('context', 'dict'), ('values', 'numpy.ndarray'), ('coefficients', 'numpy.ndarray')]),
        node('pair_sum', 'Add caller-supplied numeric rows in their declared order', generic,
            [('values', 'numpy.ndarray'), ('coefficients', 'numpy.ndarray')], [('paired', 'numpy.ndarray')]),
        node('final_inputs', 'Supply half of the completed pair, then B4 and CropNet', prefix+'final_inputs',
            [('context', 'dict'), ('paired', 'numpy.ndarray')], [('values', 'numpy.ndarray'), ('coefficients', 'numpy.ndarray')]),
        node('final_sum', 'Combine caller-supplied numeric rows in their declared order', generic,
            [('values', 'numpy.ndarray'), ('coefficients', 'numpy.ndarray')], [('scores', 'numpy.ndarray')]),
        node('result', 'Attach source-shaped predictions to complete training evidence', prefix+'result',
            [('training_result', 'dict'), ('context', 'dict'), ('scores', 'numpy.ndarray')], [('result', 'dict')])]
    links = [('fold_plan', 'train_families', 'plan', 'object'),
        ('train_families', 'pair_inputs', 'training_result', 'dict'),
        ('pair_inputs', 'pair_sum', 'values', 'numpy.ndarray'),
        ('pair_inputs', 'pair_sum', 'coefficients', 'numpy.ndarray'),
        ('pair_inputs', 'final_inputs', 'context', 'dict'),
        ('pair_sum', 'final_inputs', 'paired', 'numpy.ndarray'),
        ('final_inputs', 'final_sum', 'values', 'numpy.ndarray'),
        ('final_inputs', 'final_sum', 'coefficients', 'numpy.ndarray'),
        ('train_families', 'result', 'training_result', 'dict'),
        ('pair_inputs', 'result', 'context', 'dict'),
        ('final_sum', 'result', 'scores', 'numpy.ndarray')]
    graph.edges = [DependencyEdge(source_id=s, target_id=t, output_name=p, input_name=p,
        source_type=k, target_type=k) for s, t, p, k in links]
    graph.metadata.update(num_nodes=7, num_edges=11,
        catalog_output_sources={'result': dict(node_id='result', output_name='result')},
        reuse_boundary='Two instances of the same generic ordered-sum provider preserve source parenthesization. Identities, five-class shape, model families, coefficients and interpretation remain recipe policy.',
        qualification_status='Full four-family execution, provider eligibility and original intake publication remain pending.')
    return graph

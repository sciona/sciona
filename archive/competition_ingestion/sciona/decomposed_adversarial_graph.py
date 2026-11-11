"""Full source attack with an explicit reusable normalization dependency."""
from sciona.adversarial_graph import build_adversarial_graph
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus


def build_decomposed_adversarial_graph():
    graph = build_adversarial_graph()
    execute = graph.nodes[1]
    execute.matched_primitive = 'sciona.atoms.dl.adversarial_normalized_execution.execute_with_normalizer'
    execute.inputs.append(IOSpec(name='normalizer', type_desc='object'))
    graph.nodes.insert(1, AlgorithmicNode(
        node_id='normalize_rows', name='Normalize rows by explicit scale',
        description='Reusable uncentered row scaling invoked within each source momentum update.',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.numerical.statistics.row_scale.normalize_rows_by_scale',
        inputs=[IOSpec(name='values', type_desc='numpy.ndarray'), IOSpec(name='statistic', type_desc='str')],
        outputs=[IOSpec(name='normalized', type_desc='numpy.ndarray')]))
    graph.edges.append(DependencyEdge(source_id='normalize_rows', target_id='execute',
        output_name='normalized', input_name='normalizer', source_type='object', target_type='object',
        edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(num_nodes=3, num_edges=2,
        decomposition='Registered row scaling is passed as a callable and invoked for every gradient and, in targeted branches, every accumulated momentum. Source recurrence and adaptive schedules remain intact.',
        publication_status='draft')
    return graph

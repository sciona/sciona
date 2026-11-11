"""Full source recipe with four explicit reusable circular-filter nodes."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.connectomics_graph import build_connectomics_graph


def build_decomposed_connectomics_graph():
    graph = build_connectomics_graph()
    graph.nodes = [graph.nodes[0]]
    graph.edges = []
    prefix = 'sciona.atoms.ml.connectomics_filter_execution.'

    def node(identity, primitive, inputs, outputs):
        graph.nodes.append(AlgorithmicNode(
            node_id=identity, name=identity, description=primitive,
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=primitive,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs]))

    def edge(source, target, output, input_name, kind):
        graph.edges.append(DependencyEdge(source_id=source, target_id=target,
            output_name=output, input_name=input_name, source_type=kind, target_type=kind))

    controls = [(f'{f}_{control}', 'object') for f in ('f1', 'f2', 'f3', 'f4')
                for control in ('shifts', 'weights')]
    node('configuration', prefix + 'filter_configuration', [('prepared', 'object')],
         [('values', 'numpy.ndarray'), ('axis', 'int')] + controls)
    edge('prepare', 'configuration', 'prepared', 'prepared', 'object')
    for f in ('f1', 'f2', 'f3', 'f4'):
        node(f, 'sciona.atoms.numerical.spatial.circular_filter.ordered_circular_filter',
             [('values', 'numpy.ndarray'), ('shifts', 'object'), ('weights', 'object'), ('axis', 'int')],
             [('filtered', 'numpy.ndarray')])
        for control, kind in [('values', 'numpy.ndarray'), ('axis', 'int'), ('shifts', 'object'), ('weights', 'object')]:
            output = f'{f}_{control}' if control in ('shifts', 'weights') else control
            edge('configuration', f, output, control, kind)
    node('execute', prefix + 'execute_prefiltered', [('prepared', 'object')] +
         [(f, 'numpy.ndarray') for f in ('f1', 'f2', 'f3', 'f4')], [('result', 'dict')])
    edge('prepare', 'execute', 'prepared', 'prepared', 'object')
    for f in ('f1', 'f2', 'f3', 'f4'):
        edge(f, 'execute', 'filtered', f, 'numpy.ndarray')
    graph.metadata.update(num_nodes=len(graph.nodes), num_edges=len(graph.edges),
        decomposition='Four generic ordered circular filters with explicit source recipe taps. Both modes compute all four filters; the simple sweep consumes only f1/f2.',
        publication_status='draft')
    return graph

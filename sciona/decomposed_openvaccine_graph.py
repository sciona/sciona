"""Complete reconstruction with separate folding, training and generic sum."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.openvaccine_graph import build_openvaccine_graph


def build_decomposed_openvaccine_graph():
    graph = build_openvaccine_graph()
    def node(name, description, inputs, outputs, fqdn=None):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_openvaccine_execution.'+name,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs])
    graph.nodes = [graph.nodes[0],
        node('features', 'Fold raw RNA populations into dense graph features',
             [('prepared', 'object')], [('featured', 'dict')]),
        node('fit_forecast', 'Run complete configured training and checkpoint ensemble from dense features',
             [('featured', 'dict')], [('forecasted', 'dict')]),
        node('blend_inputs', 'Expose clipped member predictions in their inference order',
             [('forecasted', 'dict')], [('values', 'ndarray'), ('coefficients', 'ndarray')]),
        node('blend', 'Sum numeric rows in caller order using float64 arithmetic',
             [('values', 'ndarray'), ('coefficients', 'ndarray')], [('total', 'ndarray')],
             'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'),
        node('result', 'Restore the ensemble mean and package full teacher statistics and lifecycle counters',
             [('forecasted', 'dict'), ('total', 'ndarray')], [('result', 'dict')])]
    links = [('prepare', 'features', 'prepared', 'object'),
             ('features', 'fit_forecast', 'featured', 'dict'),
             ('fit_forecast', 'blend_inputs', 'forecasted', 'dict'),
             ('fit_forecast', 'result', 'forecasted', 'dict'),
             ('blend_inputs', 'blend', 'values', 'ndarray'),
             ('blend_inputs', 'blend', 'coefficients', 'ndarray'),
             ('blend', 'result', 'total', 'ndarray')]
    graph.edges = [DependencyEdge(source_id=s, target_id=t, output_name=p, input_name=p,
                                 source_type=k, target_type=k) for s, t, p, k in links]
    graph.metadata.update(num_nodes=6, num_edges=7,
        catalog_output_sources={'result': dict(node_id='result', output_name='result')},
        reuse_boundary='Numeric ordered sum is domain independent. Dense-feature training accepts compatible caller-defined graph features with fixed 55/8/5 dimensions and pinned architectures. RNA folding, member inventory, clipping, mean division and training policy remain recipe-specific.',
        qualification_status='Complete current execution, semantic review and original-intake approval pending.')
    return graph

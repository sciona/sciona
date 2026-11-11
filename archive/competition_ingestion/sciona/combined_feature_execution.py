"""Source combined-feature model from raw signal segments to segment probabilities."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.relative_power_execution import build_relative_power_execution_graph


def build_combined_feature_execution_graph():
    from sciona.atoms.riemannian_bci.signal_processing.combined_feature_inputs import CONFIG_PORTS
    graph = build_relative_power_execution_graph()
    array = 'numpy.ndarray'
    signal = 'sciona.atoms.riemannian_bci.signal_processing.'
    config = next(n for n in graph.nodes if n.node_id == 'configuration')
    config.matched_primitive = signal + 'combined_feature_inputs.combined_feature_configuration'
    config.outputs = [IOSpec(name=name, type_desc=kind) for name, kind in CONFIG_PORTS]
    config.description = 'Expose source combined-feature settings, including five bags and AR order/subsampling.'
    graph.edges = [e for e in graph.edges if e.target_id != 'vectorize']
    def edge(source, output, target, port, kind=array):
        graph.edges.append(DependencyEdge(source_id=source, target_id=target, output_name=output, input_name=port, source_type=kind, target_type=kind))
    families = [('ar_standard_errors', 'ar_standard_errors.channel_ar_standard_errors', [('order', 'int'), ('subsample', 'int')]),
                ('basic_statistics', 'basic_statistics.channel_basic_statistics', []),
                ('fractal_features', 'fractal_features.channel_fractal_features', [])]
    for part in ['training', 'prediction']:
        for family, primitive, settings in families:
            identity = part + '_' + family
            graph.nodes.append(AlgorithmicNode(node_id=identity, name=identity.replace('_', ' '),
                description='Compute the source feature family independently on the common ordered windows.',
                concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=signal + primitive,
                inputs=[IOSpec(name='windows', type_desc=array)] + [IOSpec(name=n, type_desc=t) for n, t in settings],
                outputs=[IOSpec(name='feature_tensor', type_desc=array)]))
            edge('windows', part + '_windows', identity, 'windows')
            for name, kind in settings:
                edge('configuration', name, identity, name, kind)
        combined = part + '_combined'
        graph.nodes.append(AlgorithmicNode(node_id=combined, name=combined.replace('_', ' '),
            description='Concatenate source feature families within each channel before flattening.', concept_type='custom',
            status=NodeStatus.ATOMIC, matched_primitive=signal + 'feature_concatenation.concatenate_channel_features',
            inputs=[IOSpec(name=n, type_desc=array) for n in ['relative_power', 'ar_standard_errors', 'basic_statistics', 'fractal_features']],
            outputs=[IOSpec(name='combined_tensor', type_desc=array)]))
        edge(part + '_power', 'power_tensor', combined, 'relative_power')
        for family, _, _ in families:
            edge(part + '_' + family, 'feature_tensor', combined, family)
        edge(combined, 'combined_tensor', 'vectorize', part + '_tensor')
    next(n for n in graph.nodes if n.node_id == 'segments').outputs[0].name = 'combined_feature_segment_probabilities'
    graph.metadata.update(scope='Single source combined-feature model: relative power, AR coefficient standard errors, basic statistics and PFD/HFD/Hurst, five bagged classifiers, segment max. Complete eleven-model ensemble is outside scope.',
                          num_nodes=len(graph.nodes), num_edges=len(graph.edges))
    return graph

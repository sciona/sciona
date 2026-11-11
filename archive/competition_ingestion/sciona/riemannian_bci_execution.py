"""Executable source-configured pair of Riemannian competition model branches."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_riemannian_branch_graph():
    """Raw segments and binary training labels to two aligned segment score vectors.

    This reconstructs two of eleven source ensemble branches. Outputs remain
    separate: averaging only these two would not reproduce the complete blend.
    """
    from sciona.atoms.riemannian_bci.signal_processing.source_inputs import CONFIG_PORTS
    array = 'numpy.ndarray'
    signal = 'sciona.atoms.riemannian_bci.'
    nodes, edges = [], []

    def node(identity, primitive, inputs, outputs, description):
        nodes.append(AlgorithmicNode(node_id=identity, name=identity.replace('_', ' '),
                                    description=description, concept_type='custom', status=NodeStatus.ATOMIC,
                                    matched_primitive=primitive,
                                    inputs=[IOSpec(name=name, type_desc=kind) for name, kind in inputs],
                                    outputs=[IOSpec(name=name, type_desc=kind) for name, kind in outputs]))

    def edge(source, output, target, port, kind=array):
        edges.append(DependencyEdge(source_id=source, target_id=target, output_name=output,
                                    input_name=port, source_type=kind, target_type=kind))

    def config(output, target, port=None):
        edge('configuration', output, target, port or output, dict(CONFIG_PORTS)[output])

    node('configuration', signal + 'signal_processing.source_inputs.source_riemann_configuration', [], CONFIG_PORTS,
         'Expose source YAML settings with distinct ports for each branch.')
    node('windows', signal + 'signal_processing.source_inputs.prepare_segment_windows',
         [('training_segments', 'list'), ('prediction_segments', 'list'), ('window_size', 'int'), ('hop_size', 'int')],
         [('training_windows', array), ('training_indices', array), ('prediction_windows', array),
          ('prediction_indices', array), ('n_segments', 'int')], 'Frame each segment independently and retain segment identity.')
    for name in ['window_size', 'hop_size']:
        config(name, 'windows')
    for branch in ['autocorrelation', 'coherence']:
        if branch == 'autocorrelation':
            primitive = signal + 'covariance_features.delay_correlation.channel_delay_correlations'
            settings = [('delays', 'list'), ('subsample', 'int')]
        else:
            primitive = signal + 'covariance_features.frequency_coherence.frequency_band_coherence'
            settings = [('frequency_bands', 'list'), ('fs', 'float'), ('fft_window', 'int'), ('overlap', 'float')]
        for part in ['training', 'prediction']:
            identity = branch + '_' + part
            node(identity, primitive, [('windows', array)] + settings, [('matrices', array)],
                 'Compute source-compatible per-channel matrix features for this partition.')
            edge('windows', part + '_windows', identity, 'windows')
            for name, _ in settings:
                config(name, identity)
        tangent = branch + '_tangent'
        node(tangent, signal + 'covariance_features.source_tangent.channel_tangent_features',
             [('training_matrices', array), ('prediction_matrices', array), ('metric', 'str'), ('tsupdate', 'bool')],
             [('training_features', array), ('prediction_features', array)], 'Apply source shrinkage and branch-specific tangent reference behavior.')
        for part in ['training', 'prediction']:
            edge(branch + '_' + part, 'matrices', tangent, part + '_matrices')
        config(branch + '_metric', tangent, 'metric')
        config(branch + '_update', tangent, 'tsupdate')
        classifier = branch + '_classifier'
        node(classifier, 'sciona.atoms.ml.xgboost.competition_bagging.bagged_window_probabilities',
             [('training_features', array), ('training_segment_indices', array), ('segment_labels', array),
              ('prediction_features', array), ('n_bags', 'int'), ('n_estimators', 'int')],
             [('probabilities', array)], 'Fit the source-configured bagged classifier and score prediction windows.')
        for part in ['training', 'prediction']:
            edge(tangent, part + '_features', classifier, part + '_features')
        edge('windows', 'training_indices', classifier, 'training_segment_indices')
        config(branch + '_bags', classifier, 'n_bags')
        config('n_estimators', classifier)
        reduction = branch + '_segments'
        node(reduction, signal + 'signal_processing.segment_scores.segment_probability_max',
             [('window_probabilities', array), ('segment_indices', array), ('n_segments', 'int')],
             [(branch + '_segment_probabilities', array)], 'Return one maximum probability for each prediction segment in input order.')
        edge(classifier, 'probabilities', reduction, 'window_probabilities')
        edge('windows', 'prediction_indices', reduction, 'segment_indices')
        edge('windows', 'n_segments', reduction, 'n_segments', 'int')
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'artifact_source': 'competition_execution_reconstruction', 'publication_status': 'draft',
        'scope': 'Two source Riemannian model branches from raw segments through fitting and segment probabilities; nine other ensemble branches and final eleven-model blend remain outside this component.',
        'num_nodes': len(nodes), 'num_edges': len(edges), 'trust_tier': 3,
        'source_revision': '00f937cc7710977dc812d9fc675864e2b8288658',
        'source_configuration_fixed': True,
        'limitations': 'Modern XGBoost/sklearn execution; original training engine and competition performance not reproduced.',
    })

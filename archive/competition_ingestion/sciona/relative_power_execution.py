"""Source-configured relative-log-power branch from raw segments to scores."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_relative_power_execution_graph():
    from sciona.atoms.riemannian_bci.signal_processing.relative_power_inputs import CONFIG_PORTS
    array = 'numpy.ndarray'
    signal = 'sciona.atoms.riemannian_bci.'
    nodes, edges = [], []
    def node(identity, primitive, inputs, outputs, description):
        nodes.append(AlgorithmicNode(node_id=identity, name=identity.replace('_', ' '), description=description,
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=primitive,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs], outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs]))
    def edge(source, output, target, port, kind=array):
        edges.append(DependencyEdge(source_id=source, target_id=target, output_name=output, input_name=port, source_type=kind, target_type=kind))
    def config(name, target):
        edge('configuration', name, target, name, dict(CONFIG_PORTS)[name])
    node('configuration', signal + 'signal_processing.relative_power_inputs.relative_power_configuration', [], CONFIG_PORTS,
         'Expose source feature/model settings as distinct immutable graph configuration ports.')
    node('windows', signal + 'signal_processing.source_inputs.prepare_segment_windows',
         [('training_segments', 'list'), ('prediction_segments', 'list'), ('window_size', 'int'), ('hop_size', 'int')],
         [('training_windows', array), ('training_indices', array), ('prediction_windows', array), ('prediction_indices', array), ('n_segments', 'int')],
         'Window both partitions independently and retain segment identities.')
    for name in ['window_size', 'hop_size']:
        config(name, 'windows')
    settings = [('frequency_bands', 'list'), ('fs', 'float'), ('fft_window', 'int'), ('overlap', 'float')]
    for part in ['training', 'prediction']:
        node(part + '_power', signal + 'covariance_features.relative_power.relative_log_band_power',
             [('windows', array)] + settings, [('power_tensor', array)], 'Compute source mean-band relative log Welch powers.')
        edge('windows', part + '_windows', part + '_power', 'windows')
        for name, _ in settings:
            config(name, part + '_power')
    node('vectorize', signal + 'signal_processing.feature_vectorization.vectorize_feature_partitions',
         [('training_tensor', array), ('prediction_tensor', array)], [('training_features', array), ('prediction_features', array)],
         'Flatten non-sample axes in source order and require equal training/prediction feature shapes.')
    for part in ['training', 'prediction']:
        edge(part + '_power', 'power_tensor', 'vectorize', part + '_tensor')
    node('classifier', 'sciona.atoms.ml.xgboost.competition_bagging.bagged_window_probabilities',
         [('training_features', array), ('training_segment_indices', array), ('segment_labels', array),
          ('prediction_features', array), ('n_bags', 'int'), ('n_estimators', 'int')], [('window_probabilities', array)],
         'Fit the source-configured bagged boosted trees and score prediction windows.')
    for part in ['training', 'prediction']:
        edge('vectorize', part + '_features', 'classifier', part + '_features')
    edge('windows', 'training_indices', 'classifier', 'training_segment_indices')
    for name in ['n_bags', 'n_estimators']:
        config(name, 'classifier')
    node('segments', signal + 'signal_processing.segment_scores.segment_probability_max',
         [('window_probabilities', array), ('segment_indices', array), ('n_segments', 'int')],
         [('relative_power_segment_probabilities', array)], 'Take the maximum score per prediction segment in input order.')
    edge('classifier', 'window_probabilities', 'segments', 'window_probabilities')
    edge('windows', 'prediction_indices', 'segments', 'segment_indices')
    edge('windows', 'n_segments', 'segments', 'n_segments', 'int')
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'artifact_source': 'competition_execution_reconstruction', 'publication_status': 'draft', 'trust_tier': 3,
        'scope': 'Single relative-log-power source model branch from raw segments through training and segment probabilities; complete eleven-model ensemble is outside scope.',
        'source_revision': '00f937cc7710977dc812d9fc675864e2b8288658', 'source_configuration_fixed': True,
        'num_nodes': len(nodes), 'num_edges': len(edges),
        'limitations': 'Modern-library source execution; no historical training engine or predictive-performance claim.'})

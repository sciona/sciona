"""Source Feng base KNN branch from raw segment partitions to segment scores."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_feng_knn_execution_graph():
    nodes, edges = [], []
    signal = 'sciona.atoms.riemannian_bci.signal_processing.feng_partitions.'
    array = 'numpy.ndarray'
    def node(name, primitive, inputs, outputs, description):
        nodes.append(AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=primitive,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs], outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs]))
    node('filter', signal+'feng_filter_partitions', [('training_segments', 'list'), ('prediction_segments', 'list')],
         [('training_filtered', 'list'), ('prediction_filtered', 'list')],
         'Resample each 600-second segment to 400 Hz and apply source causal bandpass filtering independently.')
    node('features', signal+'feng_feature_partitions', [('training_filtered', 'list'), ('prediction_filtered', 'list')],
         [('training_features', array), ('prediction_features', array)],
         'Compute mean log FFT magnitude bands and standard deviation in twenty ordered windows per segment.')
    node('classifier', 'sciona.atoms.ml.sklearn.neighbors.feng_knn.feng_knn_segment_probabilities',
         [('training_features', array), ('segment_labels', array), ('prediction_features', array)],
         [('feng_knn_segment_probabilities', array)],
         'Fit separately scaled source KNN partitions and average positive-class window probabilities per segment.')
    for part in ['training', 'prediction']:
        for source, target, suffix, kind in [('filter', 'features', 'filtered', 'list'), ('features', 'classifier', 'features', array)]:
            port = part+'_'+suffix
            edges.append(DependencyEdge(source_id=source, target_id=target, output_name=port, input_name=port, source_type=kind, target_type=kind))
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'artifact_source': 'competition_execution_reconstruction', 'publication_status': 'draft', 'trust_tier': 3,
        'scope': 'Single Feng base KNN model for one caller-selected population; caller supplies ordered 600-second clips and binary training labels. Complete eleven-model ensemble is outside scope.',
        'source_revision': '00f937cc7710977dc812d9fc675864e2b8288658', 'source_configuration_fixed': True,
        'num_nodes': len(nodes), 'num_edges': len(edges),
        'limitations': 'Prediction-batch scaling follows source. Modern numerical libraries; no historical environment, clinical or predictive-performance claim.'})

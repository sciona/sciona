"""Source expanded-feature Feng KNN and GLM branches with shared preprocessing."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_feng_expanded_execution_graph():
    nodes, edges = [], []
    array = 'numpy.ndarray'
    def node(name, primitive, inputs, outputs, description):
        nodes.append(AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=primitive,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs], outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs]))
    node('filter', 'sciona.atoms.riemannian_bci.signal_processing.feng_partitions.feng_filter_partitions',
         [('training_segments', 'list'), ('prediction_segments', 'list')],
         [('training_filtered', 'list'), ('prediction_filtered', 'list')],
         'Independently resample and causally filter ordered 600-second source clips.')
    node('features', 'sciona.atoms.riemannian_bci.signal_processing.feng_expanded_partitions.feng_expanded_partitions',
         [('training_filtered', 'list'), ('prediction_filtered', 'list')],
         [('training_features', array), ('prediction_features', array)],
         'Compute 384 ordered spectral/correlation features in twelve 50-second windows per 16-channel clip.')
    for name, primitive, description in [
        ('knn', 'neighbors.feng_expanded_knn.feng_expanded_knn_segment_probabilities', 'Source forty-neighbor classifier with separate partition scalers and mean window probabilities.'),
        ('glm', 'linear_model.feng_glm.feng_glm_segment_probabilities', 'Source L2/C=.6 logistic regression with training-scaler reuse and mean window probabilities.')]:
        node(name, 'sciona.atoms.ml.sklearn.'+primitive,
             [('training_features', array), ('segment_labels', array), ('prediction_features', array)],
             [('feng_'+('expanded_knn' if name == 'knn' else 'glm')+'_segment_probabilities', array)], description)
    for part in ['training', 'prediction']:
        for source, target, suffix, kind in [('filter', 'features', 'filtered', 'list'), ('features', 'knn', 'features', array), ('features', 'glm', 'features', array)]:
            port = part+'_'+suffix
            edges.append(DependencyEdge(source_id=source, target_id=target, output_name=port, input_name=port, source_type=kind, target_type=kind))
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'artifact_source': 'competition_execution_reconstruction', 'publication_status': 'draft', 'trust_tier': 3,
        'scope': 'Two expanded-feature Feng models for one caller-selected population; ordered 600-second 16-channel clips and binary training labels required. Full eleven-model ensemble remains outside scope.',
        'source_revision': '00f937cc7710977dc812d9fc675864e2b8288658', 'source_configuration_fixed': True,
        'num_nodes': len(nodes), 'num_edges': len(edges),
        'limitations': 'KNN prediction-batch scaling and GLM training-scaler reuse follow source. Current sklearn LBFGS for unspecified GLM solver; no historical-engine or performance claim.'})

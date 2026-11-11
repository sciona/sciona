"""Three source Andriy branches sharing independent population preprocessing."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_andriy_execution_graph():
    population_outputs = ['training_features', 'training_labels',
                          'prediction_features', 'valid_prediction_rows']
    nodes = [AlgorithmicNode(
        node_id='populations', name='populations', concept_type='custom', status=NodeStatus.ATOMIC,
        description='Prepare three ordered populations independently, preserving source P/P1/I order.',
        matched_primitive='sciona.atoms.riemannian_bci.signal_processing.andriy_populations.andriy_documented_populations',
        inputs=[IOSpec(name=name, type_desc='list') for name in [
            'csp_candidates', 'candidate_sequences', 'primary_positive',
            'auxiliary_positive', 'negative', 'prediction', 'sampling_frequencies']],
        outputs=[IOSpec(name=name, type_desc='list') for name in population_outputs])]
    edges = []
    for branch, module, auxiliary in [
        ('xgb', 'xgboost', 'segment_model_probabilities'),
        ('svm', 'liblinear', 'unmasked_window_probabilities'),
        ('glm', 'glmnet', 'segment_model_probabilities'),
    ]:
        nodes.append(AlgorithmicNode(
            node_id=branch, name=branch, concept_type='custom', status=NodeStatus.ATOMIC,
            description=f'Execute source {branch} branch across three ordered populations with pinned R runtime.',
            matched_primitive=f'sciona.atoms.ml.{module}.andriy_r_{branch}.andriy_r_{branch}_segment_probabilities',
            inputs=[IOSpec(name=name, type_desc='list') for name in population_outputs],
            outputs=[IOSpec(name=f'andriy_{branch}_{name}', type_desc='numpy.ndarray')
                     for name in ['scores', auxiliary]]))
        edges.extend(DependencyEdge(source_id='populations', target_id=branch,
            output_name=name, input_name=name, source_type='list', target_type='list')
            for name in population_outputs)
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'artifact_source': 'competition_execution_reconstruction',
        'publication_status': 'draft', 'trust_tier': 3,
        'source_revision': '00f937cc7710977dc812d9fc675864e2b8288658',
        'source_configuration_fixed': True, 'num_nodes': len(nodes), 'num_edges': len(edges),
        'scope': 'Three Andriy model branches from raw clips across three explicitly ordered populations; full eleven-model ensemble remains outside this graph.',
        'limitations': 'Documented FIR/Welch/AR adaptations and pinned current R APIs; no historical engine or predictive quality claim. Caller supplies safe memberships and source enumeration order. Feature selectors require at least 300/200 used features; degenerate inputs fail closed.',
    })

"""Compose all eleven source model branches with explicit population identities."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.riemannian_bci_execution import build_riemannian_branch_graph
from sciona.combined_feature_execution import build_combined_feature_execution_graph
from sciona.relative_power_execution import build_relative_power_execution_graph
from sciona.feng_knn_execution import build_feng_knn_execution_graph
from sciona.feng_xgb_execution import build_feng_xgb_execution_graph
from sciona.feng_expanded_execution import build_feng_expanded_execution_graph
from sciona.andriy_execution import build_andriy_execution_graph


def build_ensemble_execution_graph():
    nodes, edges = [], []
    base = 'sciona.atoms.riemannian_bci.signal_processing.'
    array = 'numpy.ndarray'
    def node(name, primitive, inputs, outputs, description):
        nodes.append(AlgorithmicNode(node_id=name, name=name, concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=base+primitive, description=description,
            inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs]))
    def edge(source, output, target, port, kind=array):
        edges.append(DependencyEdge(source_id=source, output_name=output, target_id=target,
            input_name=port, source_type=kind, target_type=kind))
    names = [f'{family}_{part}_populations' for family in ['alex', 'feng']
             for part in ['training', 'prediction', 'label', 'identity']]
    node('families', 'ensemble_wiring.ensemble_family_inputs',
        [(name, 'list') for name in names+['prediction', 'andriy_identity_populations']],
        [(name, 'list') for name in names+['andriy_prediction_populations', 'andriy_identity_populations']],
        'Expose caller-supplied independent family groups and identities; no membership inference or conversion.')
    node('indices', 'ensemble_partitions.ensemble_population_indices', [],
        [(f'population{i}', 'int') for i in range(3)], 'Fixed source population positions.')
    families = [('alex', [('riemannian', build_riemannian_branch_graph),
        ('combined', build_combined_feature_execution_graph), ('relative', build_relative_power_execution_graph)]),
        ('feng', [('knn', build_feng_knn_execution_graph), ('xgb', build_feng_xgb_execution_graph),
                  ('expanded', build_feng_expanded_execution_graph)])]
    output_models = {'combined_feature_segment_probabilities': 'combined',
        'autocorrelation_segment_probabilities': 'autocorrelation', 'coherence_segment_probabilities': 'coherence',
        'relative_power_segment_probabilities': 'relative_power', 'feng_xgb_segment_probabilities': 'feng_xgb',
        'feng_knn_segment_probabilities': 'feng_knn', 'feng_expanded_knn_segment_probabilities': 'feng_expanded_knn',
        'feng_glm_segment_probabilities': 'feng_glm'}
    endpoints = {}
    for family, builders in families:
        for population in range(3):
            selector = f'{family}_population{population}'
            node(selector, 'ensemble_partitions.select_ensemble_population',
                [(part+'_populations', 'list') for part in ['training', 'prediction', 'label', 'identity']]+[('population_index', 'int')],
                [('training_segments', 'list'), ('prediction_segments', 'list'), ('segment_labels', array), ('prediction_ids', array)],
                'Select ordered clips, labels and identities for this family/population.')
            for part in ['training', 'prediction', 'label', 'identity']:
                edge('families', f'{family}_{part}_populations', selector, part+'_populations', 'list')
            edge('indices', f'population{population}', selector, 'population_index', 'int')
            for label, builder in builders:
                graph = builder()
                prefix = selector+'_'+label+'_'
                consumed = {(e.target_id, e.input_name) for e in graph.edges}
                producers = {e.source_id for e in graph.edges}
                for original in graph.nodes:
                    identity = prefix+original.node_id
                    nodes.append(original.model_copy(deep=True, update={'node_id': identity}))
                    for port in original.inputs:
                        if (original.node_id, port.name) not in consumed:
                            if port.name not in {'training_segments', 'prediction_segments', 'segment_labels'}:
                                raise ValueError('Unexpected branch root: '+port.name)
                            edge(selector, port.name, identity, port.name, port.type_desc)
                    if original.node_id not in producers:
                        for port in original.outputs:
                            model = output_models[port.name]
                            endpoints[(model, population)] = (identity, port.name, selector)
                edges.extend(e.model_copy(deep=True, update={'source_id': prefix+e.source_id,
                    'target_id': prefix+e.target_id}) for e in graph.edges)
    models = ['combined', 'autocorrelation', 'coherence', 'relative_power',
              'feng_xgb', 'feng_knn', 'feng_expanded_knn', 'feng_glm']
    if set(endpoints) != {(model, i) for model in models for i in range(3)}:
        raise ValueError('Incomplete upstream model population coverage')
    for model in models:
        collector = 'collect_'+model
        node(collector, 'ensemble_partitions.collect_population_scores',
            [(name+str(i), array) for i in range(3) for name in ['scores', 'ids']],
            [('scores', array), ('ids', array)], 'Concatenate paired population scores/IDs without ranking or sorting.')
        for i in range(3):
            identity, port, selector = endpoints[(model, i)]
            edge(identity, port, collector, 'scores'+str(i))
            edge(selector, 'prediction_ids', collector, 'ids'+str(i))
    andriy = build_andriy_execution_graph()
    for original in andriy.nodes:
        nodes.append(original.model_copy(deep=True, update={'node_id': 'andriy_'+original.node_id}))
    edges.extend(e.model_copy(deep=True, update={'source_id': 'andriy_'+e.source_id,
        'target_id': 'andriy_'+e.target_id}) for e in andriy.edges)
    edge('families', 'andriy_prediction_populations', 'andriy_populations', 'prediction', 'list')
    node('andriy_ids', 'ensemble_partitions.ordered_prediction_ids',
        [('prediction_populations', 'list'), ('identity_populations', 'list')], [('ids', array)],
        'Verify each Andriy prediction population count before concatenating IDs.')
    edge('families', 'andriy_prediction_populations', 'andriy_ids', 'prediction_populations', 'list')
    edge('families', 'andriy_identity_populations', 'andriy_ids', 'identity_populations', 'list')
    all_models = models+['andriy_svm', 'andriy_glm', 'andriy_xgb']
    node('pack', 'ensemble_wiring.pack_eleven_predictions',
        [(name, array) for name in all_models+['alex_ids', 'feng_ids', 'andriy_ids']],
        [('predictions', 'list'), ('prediction_ids', 'list')], 'Pack eleven model vectors in source blend order.')
    for model in models:
        edge('collect_'+model, 'scores', 'pack', model)
    for branch in ['svm', 'glm', 'xgb']:
        edge('andriy_'+branch, 'andriy_'+branch+'_scores', 'pack', 'andriy_'+branch)
    edge('collect_combined', 'ids', 'pack', 'alex_ids')
    edge('collect_feng_xgb', 'ids', 'pack', 'feng_ids')
    edge('andriy_ids', 'ids', 'pack', 'andriy_ids')
    node('blend', 'ensemble_alignment.aligned_eleven_model_blend',
        [('predictions', 'list'), ('prediction_ids', 'list'), ('output_ids', array), ('baseline', array)],
        [('ensemble_scores', array)], 'Rank each full model vector, align requested IDs and add equal weights to explicit baseline.')
    for name in ['predictions', 'prediction_ids']:
        edge('pack', name, 'blend', name, 'list')
    return CDGExport(nodes=nodes, edges=edges, metadata=dict(
        artifact_source='competition_execution_reconstruction', publication_status='draft', trust_tier=3,
        source_revision='00f937cc7710977dc812d9fc675864e2b8288658', source_configuration_fixed=True,
        num_nodes=len(nodes), num_edges=len(edges),
        scope='All eleven model branches with separate caller-supplied family populations and explicit identity-aligned final blend.',
        limitations='Draft composition awaiting complete execution/reference validation. Existing branch runtime/input restrictions apply; no historical-engine or predictive-quality claim.'))

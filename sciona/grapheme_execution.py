"""Complete explicit training/calibration/query graph, pending qualification."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_grapheme_graph():
    prefix = 'sciona.atoms.dl.grapheme_execution.'
    nodes, edges = [], []
    def node(identity, primitive, inputs, outputs):
        nodes.append(AlgorithmicNode(node_id=identity, name=identity.replace('_', ' '),
            description='Explicit grapheme workflow stage: '+primitive, concept_type='custom',
            status=NodeStatus.ATOMIC, matched_primitive=prefix+primitive,
            inputs=[IOSpec(name=name, type_desc='object', required=True) for name in inputs],
            outputs=[IOSpec(name=name, type_desc='object') for name in outputs]))
    def edge(source, target, output, input):
        edges.append(DependencyEdge(source_id=source, target_id=target, output_name=output,
                                    input_name=input, source_type='object', target_type='object'))
    node('inference_inputs', 'inference_inputs', ['validation_images', 'query_images'], ['validation_images', 'query_images'])
    for identity, spec in [('train_ood', 'ood_spec'), ('train_seen', 'seen_spec'),
                           ('train_font_a', 'font_a_spec'), ('train_font_b', 'font_b_spec')]:
        node(identity, identity, [spec], ['state'])
    for suffix in ('a', 'b'):
        identity = 'train_generator_'+suffix
        node(identity, identity, ['gan_'+suffix+'_spec', 'classifier_state'], ['state'])
        edge('train_font_'+suffix, identity, 'state', 'classifier_state')
    for population, images in [('validation', 'validation_images'), ('query', 'query_images')]:
        for branch in ('ood', 'seen', 'font_a', 'font_b'):
            identity = branch+'_'+population
            generator = branch.startswith('font_')
            node(identity, 'predict_generated' if generator else 'predict_b7', ['state', 'images'], ['logits'])
            source = 'train_generator_'+branch[-1] if generator else 'train_'+branch
            edge(source, identity, 'state', 'state')
            edge('inference_inputs', identity, images, 'images')
        scores, ensemble = 'scores_'+population, 'ensemble_'+population
        node(scores, 'classifier_scores', ['ood_logits', 'seen_logits'], ['scores'])
        edge('ood_'+population, scores, 'logits', 'ood_logits')
        edge('seen_'+population, scores, 'logits', 'seen_logits')
        node(ensemble, 'font_ensemble', ['logits_a', 'logits_b'], ['predictions'])
        edge('font_a_'+population, ensemble, 'logits', 'logits_a')
        edge('font_b_'+population, ensemble, 'logits', 'logits_b')
    node('calibration', 'fit_router', ['classifier_scores', 'unseen_predictions', 'truth_components',
         'known_class_mask', 'sample_weight', 'thresholds'], ['calibration'])
    edge('scores_validation', 'calibration', 'scores', 'classifier_scores')
    edge('ensemble_validation', 'calibration', 'predictions', 'unseen_predictions')
    node('routing', 'route_predictions', ['classifier_scores', 'unseen_predictions', 'calibration',
         'known_class_mask'], ['joint_classes'])
    edge('scores_query', 'routing', 'scores', 'classifier_scores')
    edge('ensemble_query', 'routing', 'predictions', 'unseen_predictions')
    edge('calibration', 'routing', 'calibration', 'calibration')
    node('submission', 'submission_components', ['joint_classes'], ['components'])
    edge('routing', 'submission', 'joint_classes', 'joint_classes')
    return CDGExport(nodes=nodes, edges=edges, metadata=dict(
        artifact_source='competition_execution_reconstruction', publication_status='draft',
        source_version_id='0c0997e0-c5f7-509c-9068-dcf00a167f56',
        scope='Two B7 fits, two font-classifier fits, two generator fits, separate calibration/query inference, weighted threshold fitting and final routing.',
        num_nodes=len(nodes), num_edges=len(edges),
        limits=['Explicit reconstructed B7 policies do not establish exact historical normalization, warmup, sampling or seen decoder.',
                'Runtime checkpoint states require the qualified local filesystem and installed providers.',
                'Graph construction alone establishes neither successful execution nor catalog approval.']))

"""A reusable calibration/application graph with explicit numerical ports."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def correction_port(name):
    """Portable numeric contracts; target units are supplied by the caller."""
    constraints = {
        'observed': 'Nonempty finite float64 vector; aligned with calibration_predictions; same target units.',
        'calibration_predictions': 'Nonempty finite float64 vector; aligned with observed and calibration_probabilities; target units.',
        'calibration_probabilities': 'Nonempty finite float64 vector in [0,1]; aligned with calibration_predictions; both strictly above and strictly below threshold groups populated; probability of underestimation.',
        'threshold': 'Finite scalar in [0,1]; dimensionless; explicit decision threshold; equality receives no correction.',
        'predictions': 'Nonempty finite float64 vector; aligned with probabilities; same target units as calibration and offsets; independent application population size allowed.',
        'probabilities': 'Nonempty finite float64 vector in [0,1]; aligned with predictions; probability of underestimation.',
        'offsets': 'Finite float64 vector of length 2 in target units; signed [median(observed-predicted | q>threshold), median(predicted-observed | q<threshold)].',
        'corrected_predictions': 'Finite float64 vector aligned with predictions, in unchanged target units; ties unchanged; no rounding or clipping.',
    }
    return IOSpec(name=name, type_desc='float' if name == 'threshold' else 'NDArray[np.float64]',
                  constraints=constraints[name])


def build_conditional_correction_graph():
    nodes = []
    for node_id, function, inputs, output in [
        ('calibrate', 'estimate_conditional_offsets', ['observed', 'calibration_predictions', 'calibration_probabilities', 'threshold'], 'offsets'),
        ('correct', 'apply_conditional_offsets', ['predictions', 'probabilities', 'offsets', 'threshold'], 'corrected_predictions'),
    ]:
        nodes.append(AlgorithmicNode(node_id=node_id, name=function.replace('_', ' '),
            description='Conditional signed median residual calibration' if node_id == 'calibrate' else 'Apply classifier-conditioned signed offsets',
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.ml.calibration.conditional_residuals.'+function,
            inputs=[correction_port(name) for name in inputs],
            outputs=[correction_port(output)]))
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='calibrate', target_id='correct',
        output_name='offsets', input_name='offsets', source_type='NDArray[np.float64]', target_type='NDArray[np.float64]')],
        metadata=dict(artifact_source='reusable_computation_reconstruction', publication_status='draft',
            source_version_ids=['7d6f7b78-1dff-5757-9326-4edad55aaa68'],
            source_content_hashes=['5a9856b1f6a4db73e7ce12809248de193b63ad06401a6e73b8322744cb7c44bd'],
            scope='Fit signed conditional median offsets on supplied calibration arrays, then apply to a separate prediction population.',
            applicability=['Regression targets with additive residuals in consistent units', 'Supplied binary underestimation probabilities and explicit decision threshold'],
            exclusions=['Complete original competition pipeline', 'Raw domain feature extraction', 'Estimator training', 'Automatic calibration split selection', 'Guaranteed error reduction', 'Submission rounding'],
            num_nodes=2, num_edges=1))

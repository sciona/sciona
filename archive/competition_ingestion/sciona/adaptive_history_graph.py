"""Reusable history statistics with explicit time and missing-value contracts."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus


def build_adaptive_history_graph():
    inputs = [
        ('event_times', 'NDArray[np.int64]', 'One-dimensional integer timestamps in caller-defined units; empty allowed; aligned with values.'),
        ('values', 'NDArray[np.float64]', 'Finite one-dimensional measurements aligned with event_times; consistent caller-defined value units.'),
        ('query_time', 'int', 'Integer query timestamp in the same units and origin as event_times.'),
        ('lookbacks', 'NDArray[np.int64]', 'Three strictly increasing positive durations [initial,sparse,empty] in event-time units.'),
        ('minimum_initial_count', 'int', 'Positive dimensionless integer; one-time widening depends on initial count only.'),
    ]
    outputs = [
        ('count', 'int', 'Nonnegative dimensionless count in the selected open past window.'),
        ('mean', 'float', 'Mean in input value units; NaN exactly when count is zero.'),
        ('standard_deviation', 'float', 'Population standard deviation (ddof=0) in input value units; NaN exactly when count is zero.'),
    ]
    node = AlgorithmicNode(node_id='statistics', name='Adaptive history statistics',
        description='Select initial, sparse or empty-history lookback once, then compute count, mean and population standard deviation.',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.calibration.adaptive_history.adaptive_history_statistics',
        inputs=[IOSpec(name=n, type_desc=t, constraints=c) for n,t,c in inputs],
        outputs=[IOSpec(name=n, type_desc=t, constraints=c) for n,t,c in outputs])
    return CDGExport(nodes=[node], edges=[], metadata=dict(artifact_source='reusable_computation_reconstruction',
        publication_status='draft', source_version_ids=['7d6f7b78-1dff-5757-9326-4edad55aaa68'],
        source_content_hashes=['5a9856b1f6a4db73e7ce12809248de193b63ad06401a6e73b8322744cb7c44bd'],
        applicability=['Aligned historical scalar measurements with explicit one-shot lookback policy.',
                       'Downstream callers that inspect count before using potentially undefined statistics.'],
        exclusions=['Complete NASA feature or competition workflow', 'Domain joins and timestamp parsing',
                    'Automatic imputation', 'Source null-value skipping', 'Repeated window expansion'],
        num_nodes=1, num_edges=0))

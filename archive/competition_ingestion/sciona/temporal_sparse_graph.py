"""Complete six-stage generic temporal sparse lifecycle with private stream inputs."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_temporal_sparse_graph():
    nodes = [AlgorithmicNode(node_id=name, name=name, description='Generic temporal sparse ' + name,
             concept_type='custom', status=NodeStatus.ATOMIC,
             matched_primitive='sciona.atoms.ml.temporal_sparse_execution.temporal_sparse_' + name,
             inputs=[IOSpec(name=inp, type_desc=it)], outputs=[IOSpec(name=out, type_desc=ot)])
             for name, inp, out, it, ot in [('prepare', 'payload', 'prepared', 'dict', 'object'),
                                         ('execute', 'prepared', 'result', 'object', 'dict')]]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='prepare', target_id='execute',
        output_name='prepared', input_name='prepared', source_type='object', target_type='object')], metadata={
        'artifact_source': 'competition_generic_execution_reconstruction', 'publication_status': 'draft',
        'source_version_ids': ['ad8e992f-1e5d-530c-9475-c9caf4a08bad'],
        'scope': 'Six-stage generic temporal sparse regression: bounded file streams, causal features, hashed CSR encoding, chronological validation, online ensemble and RMSLE postprocessing.',
        'choices': ['Private bounded JSONL inspection and immutable disk snapshots verified before fitting',
                    'Per-entity rolling and latest-timestamp means with same-time target exclusion',
                    'Fixed-width categorical hashing and eight numeric temporal features in CSR chunks',
                    'Strict chronological training/validation/query separation with configured gaps',
                    'Single-pass squared-error and Huber SGD on log1p targets; equal log-score mean',
                    'Clipped inverse transform, streamed RMSLE and explicitly bounded prediction output'],
        'exclusions': ['Delayed target availability, rolling refits or validation-label history updates',
                       'Historical winner reproduction or empirical accuracy guarantee',
                       'Large-scale throughput, clean-install or cross-platform qualification'],
        'num_nodes': 2, 'num_edges': 1})

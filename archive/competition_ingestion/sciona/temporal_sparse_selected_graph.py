"""Private bounded streaming lifecycle with registered checkpoint selection."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_selected_temporal_graph():
    constraints = {
        'payload': 'Explicit version-1 bounded private stream recipe plus max_epochs, patience and min_delta. Nonnegative targets; globally ordered streams; training, validation and query separated by configured gaps. Caller defines entity keys, time scales, capacities, hashing, clipping and model controls.',
        'prepared': 'Trusted in-process immutable preparation with private source descriptors. Reverify all snapshots and summaries before fitting; never publish or persist descriptors in shared artifacts.',
        'result': 'Ordered bounded nonnegative predictions and selection diagnostics. Validation selects checkpoints; no independent accuracy claim. Paths, source digests and records excluded.',
    }
    nodes = []
    for name, inp, itype, otype in [('prepare', 'payload', 'dict', 'object'), ('execute', 'prepared', 'object', 'dict')]:
        nodes.append(AlgorithmicNode(node_id=name, name=name, concept_type='custom', status=NodeStatus.ATOMIC,
            description='Validate private inputs' if name == 'prepare' else 'Execute causal bounded sparse training, checkpoint selection and prediction',
            matched_primitive='sciona.atoms.ml.temporal_sparse_selected_execution.' + name,
            inputs=[IOSpec(name=inp, type_desc=itype, constraints=constraints[inp])],
            outputs=[IOSpec(name='result', type_desc=otype, constraints=constraints['prepared' if name == 'prepare' else 'result'])]))
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='prepare', target_id='execute',
        output_name='result', input_name='prepared', source_type='object', target_type='object')], metadata={
        'artifact_source': 'competition_temporal_selected_execution',
        'catalog_output_sources': {'result': {'node_id': 'execute', 'output_name': 'result'}},
        'reuse_decision': 'Generic immutable validation stopping providers are independently callable. Recipe owns RMSLE, sparse SGD, time policy, capacities and nonnegative inverse transform.',
        'streaming_fusion': 'Loading snapshots, causal features, sparse encoding, selection and query prediction share one bounded snapshot lifetime; no full population feature materialization.',
        'limitations': ['Synthetic qualification only; no high-volume throughput or historical winner claim.',
            'Validation selects an epoch; it is not independent accuracy evidence.',
            'Private trusted in-process state; no portable model or arbitrary schema claim.']})

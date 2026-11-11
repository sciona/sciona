"""Explicit signal lifecycle; numeric F1 policy remains independently reusable."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_complete_biosignal_graph():
    stages = [('prepare', 'payload', 'dict', 'object'), ('fit', 'prepared', 'object', 'object'),
              ('calibrate', 'fitted', 'object', 'object'), ('predict', 'calibrated', 'object', 'dict')]
    bounds = [
        'Explicit finite version-1 signal recipe with aligned channel meaning/sample rate and disjoint training/calibration/query subjects. Both classes required in labeled populations. Caller owns units and identities.',
        'Validated immutable prepared configuration. Training-only recording-balanced scaling, dual CNN, spectral masks and class/recording-weighted BCE. Trusted in-process model state; all features held in memory.',
        'Unmodified fitted state. Mean window scores per calibration recording; reusable empirical F1 threshold, inclusive comparison and highest-threshold ties. Selection score is not independent accuracy or probability calibration.',
        'Unmodified calibrated state. Query scores and decisions without fitting, with counts and omitted trailing samples. No clinical, historical accuracy, portable state or arbitrary channel-schema claim.',
    ]
    nodes = [AlgorithmicNode(node_id=name, name=name, description=bounds[i], concept_type='custom',
        status=NodeStatus.ATOMIC, matched_primitive='sciona.atoms.ml.biosignal_complete_execution.' + name,
        inputs=[IOSpec(name=inp, type_desc=it, constraints=bounds[i])],
        outputs=[IOSpec(name='result', type_desc=ot, constraints=bounds[i])])
        for i, (name, inp, it, ot) in enumerate(stages)]
    edges = [DependencyEdge(source_id=stages[i-1][0], target_id=name, output_name='result', input_name=inp,
        source_type='object', target_type='object') for i, (name, inp, _, _) in enumerate(stages) if i]
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'artifact_source': 'biosignal_complete_execution',
        'catalog_output_sources': {'result': {'node_id': 'predict', 'output_name': 'result'}},
        'reuse_decision': 'Reuse generic empirical-F1 fit/apply providers. Subject separation, windowing, PSD, model, aggregation and clipping/metric policy belong to the signal recipe.',
        'limits': ['Synthetic qualification only; no clinical or independent accuracy claim.',
            'Training/calibration/query separation is caller-specified and checked by subject keys.',
            'Neural batches bound forward allocation, not full-population feature memory.']})

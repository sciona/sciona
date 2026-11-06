"""Complete generic NAB reconstruction with explicit reusable thresholding."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.nab_graph import build_nab_graph


def build_decomposed_nab_graph():
    graph = build_nab_graph()
    prefix = 'sciona.atoms.ml.decomposed_nab_execution.'
    def node(name, description, inputs, outputs, fqdn=None):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or prefix+name,
            inputs=[IOSpec(name=n, type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n,t in outputs])
    controls = [('values','list'), ('history','int'), ('minimum','int'),
                ('ddof','int'), ('multiplier','float')]
    graph.nodes = [graph.nodes[0],
        node('forest_scores', 'Fit complete prior-only forests and score current windows',
             [('prepared','object')], [('forest','dict')]),
        node('threshold_inputs', 'Expose history controls and recipe multiplier three',
             [('prepared','object'), ('forest','dict')], controls),
        node('thresholds', 'Compute caller-configured prior location-scale thresholds',
             controls, [('thresholds','list')],
             'sciona.atoms.numerical.statistics.prior_threshold.prior_location_scale_threshold'),
        node('detection', 'Restore warmup slots and apply strict comparison',
             [('prepared','object'), ('forest','dict'), ('thresholds','list')], [('detected','dict')]),
        node('result', 'Retain complete detection and corrected benchmark evaluation',
             [('prepared','object'), ('detected','dict')], [('result','dict')])]
    links = [('prepare', target, 'prepared', 'object')
             for target in ('forest_scores','threshold_inputs','detection','result')]
    links += [('forest_scores', target, 'forest', 'dict') for target in ('threshold_inputs','detection')]
    links += [('threshold_inputs', 'thresholds', name, kind) for name,kind in controls]
    links += [('thresholds','detection','thresholds','list'), ('detection','result','detected','dict')]
    graph.edges = [DependencyEdge(source_id=s, target_id=t, output_name=p, input_name=p,
                                 source_type=k, target_type=k) for s,t,p,k in links]
    graph.metadata.update(num_nodes=len(graph.nodes), num_edges=len(graph.edges),
        catalog_output_sources={'result':dict(node_id='result', output_name='result')},
        reuse_boundary='Rolling prior mean-plus-scale threshold is domain independent; forest configuration, three-sigma choice, warmup restoration, strict comparison and benchmark scoring remain explicit recipe policy.',
        qualification_status='Current full graph execution and original intake approval remain pending.')
    return graph

"""Complete Santander graph with independently reusable ordered combination."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.santander_graph import build_santander_graph


def build_decomposed_santander_graph():
    graph = build_santander_graph()
    def node(name, description, inputs, outputs, fqdn=None):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_santander_execution.' + name,
            inputs=[IOSpec(name=n, type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n,t in outputs])
    graph.nodes = [graph.nodes[0],
        node('initial', 'Train all initial neural folds and retain query ranks',
             [('prepared','object')], [('initial_models','dict')]),
        node('selections', 'Validate both full hard pseudo-label tails and expand populations',
             [('prepared','object'),('initial_models','dict')], [('selected','dict')]),
        node('branches', 'Regenerate branch features and train all neural and tree refits',
             [('prepared','object'),('selected','dict')], [('fitted','dict')]),
        node('blend_inputs', 'Rank both branch outputs with average ties and expose recipe weights',
             [('fitted','dict')], [('values','ndarray'),('coefficients','ndarray')]),
        node('blend', 'Combine numeric rows in caller order without normalization',
             [('values','ndarray'),('coefficients','ndarray')], [('weighted','ndarray')],
             'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'),
        node('result', 'Normalize by recipe divisor and retain complete selection and model counts',
             [('prepared','object'),('initial_models','dict'),('selected','dict'),('fitted','dict'),('weighted','ndarray')], [('result','dict')])]
    links = [('prepare', t, 'prepared', 'object') for t in ('initial','selections','branches','result')]
    links += [('initial', t, 'initial_models', 'dict') for t in ('selections','result')]
    links += [('selections', t, 'selected', 'dict') for t in ('branches','result')]
    links += [('branches', t, 'fitted', 'dict') for t in ('blend_inputs','result')]
    links += [('blend_inputs','blend','values','ndarray'),('blend_inputs','blend','coefficients','ndarray'),
              ('blend','result','weighted','ndarray')]
    graph.edges = [DependencyEdge(source_id=s, target_id=t, output_name=p, input_name=p,
        source_type=k, target_type=k) for s,t,p,k in links]
    graph.metadata.update(num_nodes=len(graph.nodes), num_edges=len(graph.edges),
        catalog_output_sources={'result': dict(node_id='result', output_name='result')},
        reuse_boundary='Ordered weighted sum is domain independent. Pseudo tails, branch fitting, average-tie ranking, weights and post-sum divisor remain explicit recipe policy.',
        qualification_status='Full current execution and original-intake approval pending.')
    return graph

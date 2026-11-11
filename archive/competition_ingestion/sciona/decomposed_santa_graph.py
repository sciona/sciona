"""Complete source-corrected Santa recipe with reusable ordered combination."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.santa_graph import build_santa_graph


def build_decomposed_santa_graph():
    graph = build_santa_graph()
    def node(name, description, inputs, outputs, fqdn=None):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_santa_execution.' + name,
            inputs=[IOSpec(name=n, type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n,t in outputs])
    graph.nodes = [graph.nodes[0],
        node('fit', 'Generate causal replay rows and fit both complete threshold regressors',
             [('prepared','object')], [('models','object')]),
        node('predictions', 'Predict both families using held-out-selected iterations',
             [('prepared','object'), ('models','object')], [('predicted','dict')]),
        node('blend_inputs', 'Expose inverse-target values and sigmoid recipe coefficients',
             [('prepared','object'), ('predicted','dict')], [('values','ndarray'), ('coefficients','ndarray')]),
        node('blend', 'Combine caller-supplied numeric rows in order without normalization',
             [('values','ndarray'), ('coefficients','ndarray')], [('initial','ndarray')],
             'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'),
        node('result', 'Apply shared decay, seeded selection and retain full fit evidence',
             [('prepared','object'), ('models','object'), ('initial','ndarray')], [('result','dict')])]
    links = [('prepare', t, 'prepared', 'object') for t in ('fit','predictions','blend_inputs','result')]
    links += [('fit', t, 'models', 'object') for t in ('predictions','result')]
    links += [('predictions','blend_inputs','predicted','dict'),
              ('blend_inputs','blend','values','ndarray'), ('blend_inputs','blend','coefficients','ndarray'),
              ('blend','result','initial','ndarray')]
    graph.edges = [DependencyEdge(source_id=s, target_id=t, output_name=p, input_name=p,
                                 source_type=k, target_type=k) for s,t,p,k in links]
    graph.metadata.update(num_nodes=len(graph.nodes), num_edges=len(graph.edges),
        catalog_output_sources={'result':dict(node_id='result', output_name='result')},
        reuse_boundary='Ordered weighted numeric combination is domain independent. Replay sampling, two regressors, inverse target, sigmoid schedule, shared decay and action choice remain explicit recipe policy.',
        qualification_status='Current full graph qualification and original intake approval remain pending.')
    return graph

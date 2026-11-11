"""Complete Porto recipe with reusable ordered neural-probability reduction."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.porto_graph import build_porto_graph


def build_decomposed_porto_graph():
    graph = build_porto_graph()
    def node(name, description, inputs, outputs, fqdn=None):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_porto_execution.'+name,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs])
    graph.nodes = [graph.nodes[0],
        node('features','Prepare joint features with explicit column roles and RankGauss',
            [('prepared','object')],[('transformed','dict')]),
        node('fit','Fit all five DAE/classifier branches and the raw-prepared tree',
            [('prepared','object'),('transformed','dict')],[('fitted','dict')]),
        node('blend_inputs','Validate five neural probabilities and expose equal coefficients',
            [('fitted','dict')],[('values','ndarray'),('coefficients','ndarray')]),
        node('blend','Add numeric rows in caller order using float64',
            [('values','ndarray'),('coefficients','ndarray')],[('neural_sum','ndarray')],
            'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'),
        node('result','Add the tree probability, divide by six and retain full fit counts',
            [('prepared','object'),('fitted','dict'),('neural_sum','ndarray')],[('result','dict')])]
    links = [('prepare',t,'prepared','object') for t in ('features','fit','result')]
    links += [('features','fit','transformed','dict')]
    links += [('fit',t,'fitted','dict') for t in ('blend_inputs','result')]
    links += [('blend_inputs','blend','values','ndarray'),('blend_inputs','blend','coefficients','ndarray'),
              ('blend','result','neural_sum','ndarray')]
    graph.edges = [DependencyEdge(source_id=s,target_id=t,output_name=p,input_name=p,source_type=k,target_type=k)
                   for s,t,p,k in links]
    graph.metadata.update(num_nodes=len(graph.nodes),num_edges=len(graph.edges),
        catalog_output_sources={'result':dict(node_id='result',output_name='result')},
        reuse_boundary='Ordered numeric sum is domain independent. Five neural rows, unit coefficients, tree addition and division by six remain explicit recipe policy.',
        qualification_status='Full current execution and original-intake approval pending.')
    return graph

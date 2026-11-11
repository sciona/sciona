"""Full pooled forecasting recipe with a domain-independent numeric sum."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.m5_graph import build_m5_graph


def build_decomposed_m5_graph():
    graph = build_m5_graph()
    def node(name, description, inputs, outputs, fqdn=None):
        return AlgorithmicNode(node_id=name,name=name,description=description,
            concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_m5_execution.'+name,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs])
    graph.nodes = [graph.nodes[0],
        node('features','Construct all semantic-role features and full forecast grid',
             [('prepared','object')],[('transformed','dict')]),
        node('fit','Train every pool in all six families at full configured rounds',
             [('prepared','object'),('transformed','dict')],[('fitted','dict')]),
        node('forecast','Forecast every family over 28 days with separate recursive feedback',
             [('transformed','dict'),('fitted','dict')],[('forecasted','dict')]),
        node('blend_inputs','Flatten six aligned family matrices and expose unit coefficients',
             [('forecasted','dict')],[('values','ndarray'),('coefficients','ndarray')]),
        node('blend','Sum numeric rows in caller order with float64 arithmetic',
             [('values','ndarray'),('coefficients','ndarray')],[('total','ndarray')],
             'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'),
        node('result','Divide summed forecasts by six and retain complete model counts',
             [('prepared','object'),('forecasted','dict'),('total','ndarray')],[('result','dict')])]
    links = [('prepare',t,'prepared','object') for t in ('features','fit','result')]
    links += [('features',t,'transformed','dict') for t in ('fit','forecast')]
    links += [('fit','forecast','fitted','dict')]
    links += [('forecast',t,'forecasted','dict') for t in ('blend_inputs','result')]
    links += [('blend_inputs','blend',p,'ndarray') for p in ('values','coefficients')]
    links += [('blend','result','total','ndarray')]
    graph.edges = [DependencyEdge(source_id=s,target_id=t,output_name=p,input_name=p,source_type=k,target_type=k)
                   for s,t,p,k in links]
    graph.metadata.update(num_nodes=len(graph.nodes),num_edges=len(graph.edges),
        catalog_output_sources={'result':dict(node_id='result',output_name='result')},
        reuse_boundary='Ordered numeric sum is domain independent. Six families, horizon shape and division by six remain recipe policy.',
        qualification_status='Full current execution and original-intake approval pending.')
    return graph

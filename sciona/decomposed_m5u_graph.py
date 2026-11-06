"""Complete uncertainty recipe with an independently bound generic numeric sum."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.m5u_graph import build_graph


def build_decomposed_m5u_graph():
    graph=build_graph()
    def node(name,description,inputs,outputs,fqdn=None):
        return AlgorithmicNode(node_id=name,name=name,description=description,concept_type='custom',
            status=NodeStatus.ATOMIC,matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_m5u_execution.'+name,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs])
    graph.nodes=[graph.nodes[0],
        node('fit_forecast','Train every level and perform full repeated quantile inference',
             [('prepared','object')],[('forecasted','dict')]),
        node('restoration_inputs','Restore native units and expose ordered level median totals',
             [('forecasted','dict')],[('state','dict'),('values','ndarray'),('coefficients','ndarray')]),
        node('blend','Sum numeric rows in caller order using float64 arithmetic',
             [('values','ndarray'),('coefficients','ndarray')],[('total','ndarray')],
             'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'),
        node('restore','Compute the mean target and apply recipe level corrections',
             [('state','dict'),('total','ndarray')],[('restored','ndarray')]),
        node('result','Package every quantile, horizon and hierarchy series',
             [('forecasted','dict'),('restored','ndarray')],[('result','dict')])]
    links=[('prepare','fit_forecast','prepared','object')]
    links += [('fit_forecast',t,'forecasted','dict') for t in ('restoration_inputs','result')]
    links += [('restoration_inputs','blend',p,'ndarray') for p in ('values','coefficients')]
    links += [('restoration_inputs','restore','state','dict'),('blend','restore','total','ndarray'),
              ('restore','result','restored','ndarray')]
    graph.edges=[DependencyEdge(source_id=s,target_id=t,output_name=p,input_name=p,source_type=k,target_type=k)
                 for s,t,p,k in links]
    graph.metadata.update(num_nodes=len(graph.nodes),num_edges=len(graph.edges),
        catalog_output_sources={'result':dict(node_id='result',output_name='result')},
        reuse_boundary='Ordered numeric sum is domain independent. Quantiles, level selection, unit restoration, mean division and correction strength remain recipe policy.',
        qualification_status='Complete current execution and original-intake approval pending.')
    return graph

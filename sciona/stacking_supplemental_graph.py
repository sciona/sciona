"""Draft supplemental CDG exposing distances, cluster fit/apply and counts."""
import inspect

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_supplemental_graph():
    import sciona.atoms.ml.stacking_supplemental as provider
    nodes,edges=[],[]
    def node(name,function,outputs):
        fn=getattr(provider,function)
        nodes.append(AlgorithmicNode(node_id=name,name=name,description=(fn.__doc__ or function).strip(),
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=fn.__module__+'.'+fn.__name__,
            inputs=[IOSpec(name=k,type_desc=p.annotation.__name__) for k,p in inspect.signature(fn).parameters.items()],
            outputs=[IOSpec(name=k,type_desc=t) for k,t in outputs]))
    def edge(source,output,target,port):
        edges.append(DependencyEdge(source_id=source,output_name=output,target_id=target,input_name=port,
                                    source_type='object',target_type='object'))
    node('prepare','prepare_supplemental',[('inputs','object')])
    node('raw_distances','raw_class_distances',[(name,'object') for name in ('raw_one','raw_two','raw_four')])
    for name,function,output in (
        ('weighted_distances','weighted_class_distances','weighted'),
        ('embedded_distances','embedded_class_distances','embedded'),
        ('fit_clusters','fit_reference_clusters','model'),
        ('assign_clusters','cluster_assignments','clusters'),
        ('counts','nonzero_counts','counts'),('raw_neural','raw_neural_features','raw_neural')):
        node(name,function,[(output,'object')])
    for name in ('raw_distances','weighted_distances','embedded_distances','fit_clusters','assign_clusters','counts','raw_neural'):
        edge('prepare','inputs',name,'inputs')
    edge('fit_clusters','model','assign_clusters','model')
    node('collect','collect_supplemental',[('supplemental','dict'),('raw_neural','object')])
    for source,output in (
        ('raw_distances','raw_one'),('raw_distances','raw_two'),('raw_distances','raw_four'),
        ('weighted_distances','weighted'),('embedded_distances','embedded'),
        ('assign_clusters','clusters'),('counts','counts'),('raw_neural','raw_neural')):
        edge(source,output,'collect',output)
    return CDGExport(nodes=nodes,edges=edges,metadata={
        'publication_status':'draft','num_nodes':len(nodes),'num_edges':len(edges),
        'source_version_ids':['dc4969f7-e5a4-5644-a9d0-fa10493458ed'],
        'scope':'Seven supplemental blocks plus raw neural features, with separate reference cluster fitting and assignment.',
        'limitations':['Nine-class five-fold source adapters; supplied fixed embedding retains its declared population scope.',
                       'First-level models and meta learners are separate graphs; no complete intake approval.',
                       'Stage-kind witnesses do not establish static shape correctness or independent generalization.']})

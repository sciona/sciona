"""Complete draft Otto composition with one validated payload boundary."""
import inspect

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.stacking_first_graph import build_first_graph
from sciona.stacking_supplemental_graph import build_supplemental_graph
from sciona.stacking_meta_graph import build_meta_graph


def build_complete_graph(knn_groups=None):
    import sciona.atoms.ml.stacking_complete as provider
    first=build_first_graph(knn_groups)
    groups=first.metadata['knn_groups']
    nodes,edges=[],[]
    for prefix,graph in (('first',first),('supplemental',build_supplemental_graph()),('meta',build_meta_graph())):
        nodes.extend(n.model_copy(deep=True,update={'node_id':prefix+'_'+n.node_id,'name':prefix+'_'+n.name}) for n in graph.nodes)
        edges.extend(e.model_copy(deep=True,update={'source_id':prefix+'_'+e.source_id,'target_id':prefix+'_'+e.target_id}) for e in graph.edges)

    def node(name,function,outputs):
        fn=getattr(provider,function)
        nodes.append(AlgorithmicNode(node_id=name,name=name,description=(fn.__doc__ or function).strip(),
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=fn.__module__+'.'+fn.__name__,
            inputs=[IOSpec(name=k,type_desc=p.annotation.__name__) for k,p in inspect.signature(fn).parameters.items()],
            outputs=[IOSpec(name=k,type_desc=t) for k,t in outputs]))

    def wire(source,output,target,port):
        source_node=next(n for n in nodes if n.node_id==source)
        target_node=next(n for n in nodes if n.node_id==target)
        source_type=next(p.type_desc for p in source_node.outputs if p.name==output)
        target_type=next(p.type_desc for p in target_node.inputs if p.name==port)
        if source_type!=target_type:raise ValueError('Composed port type mismatch')
        edges.append(DependencyEdge(source_id=source,output_name=output,target_id=target,input_name=port,
                                    source_type=source_type,target_type=target_type))

    node('unpack','unpack_payload',provider.UNPACK_PORTS)
    node('control_gate','validate_neighbor_sharing',[('first_controls','dict')])
    node('embedding','fit_population_embedding',[('embedding','object')])
    node('layout','assemble_layout',[('layout','dict')])
    wire('unpack','first_controls','control_gate','first_controls')
    for group in groups:
        for entry in group:
            wire('first_policy',f'entry_{group[0]}','control_gate',f'representative_{entry}')
    for port in ('training','query','training_ids','query_ids','seed','embedding_controls'):
        wire('unpack',port,'embedding',port)
    for target in ('first_prepare','supplemental_prepare'):
        for port in ('training','labels','training_ids','query','query_ids','seed'):
            wire('unpack',port,target,port)
        wire('unpack','first_folds',target,'folds')
        wire('embedding','embedding',target,'embedding')
    wire('control_gate','first_controls','first_prepare','controls')
    for port in ('raw_metrics','tfidf_metrics','embedding_metrics','tfidf_controls','cluster_controls'):
        wire('unpack',port,'supplemental_prepare',port)
    wire('first_collect','first_level','layout','first_level')
    wire('supplemental_collect','supplemental','layout','supplemental')
    wire('supplemental_collect','raw_neural','layout','raw_neural')
    for port in ('training_ids','query_ids','tsne_interpretation'):
        wire('unpack',port,'layout',port)
    wire('unpack','first_folds','layout','folds')
    wire('layout','layout','meta_prepare','layout')
    for port in ('labels','seed','candidates'):
        wire('unpack',port,'meta_prepare',port)
    wire('unpack','training_ids','meta_prepare','identities')
    wire('unpack','meta_folds','meta_prepare','folds')
    # A missed edge must never become an accidental public input.
    fed=[(e.target_id,e.input_name) for e in edges]
    if len(fed)!=len(set(fed)):raise ValueError('Multiple producers for one composed input')
    roots={(n.node_id,p.name) for n in nodes for p in n.inputs if (n.node_id,p.name) not in fed}
    if roots!={('unpack','payload')}:raise ValueError('Unexpected composed graph root inputs')
    return CDGExport(nodes=nodes,edges=edges,metadata={
        'publication_status':'draft','num_nodes':len(nodes),'num_edges':len(edges),'knn_groups':groups,
        'source_version_ids':['dc4969f7-e5a4-5644-a9d0-fa10493458ed'],
        'scope':'Complete independent CPU source realization with explicit embedding, all 33 first-level entries, supplemental producers and complete meta selection/refits/blend.',
        'limitations':['Nine classes, five-fold first-level and separate four-fold meta selection; historical equivalence and winning performance unproven.',
                       'Fixed transductive embedding has no out-of-sample transform.',
                       'Five-feature embedding interpretation yields 293 first-level columns; historical 297-column ambiguity remains.',
                       'Runtime validation and in-process execution only; no static shape proof, persistent model serving or catalog approval.']})

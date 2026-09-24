"""Full single-fit training graph with score acceptance and residual outputs."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.nasa_first_fit_graph import build_nasa_first_fit_graph


def build_fit_review_graph(*, residual):
    if type(residual) is not bool:raise ValueError('Explicit residual mode required')
    nodes,edges=[],[]
    def p(name,typ):return IOSpec(name=name,type_desc=typ,constraints='Materialized validation value; aligned rows and common target units, finite scores, explicit source policy.')
    def add(identity,runtime,inputs,outputs):
        nodes.append(AlgorithmicNode(node_id=identity,name=identity,description='Explicit fit validation or feature-importance step',
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=runtime,
            inputs=[p(*v) for v in inputs],outputs=[p(*v) for v in outputs]))
    def edge(source,output,target,name):
        before=next(p for n in nodes if n.node_id==source for p in n.outputs if p.name==output)
        after=next(p for n in nodes if n.node_id==target for p in n.inputs if p.name==name)
        edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=name,
                                    source_type=before.type_desc,target_type=after.type_desc))
    base='sciona.atoms.ml.'
    add('score_policy',base+'domain_adapters.first_place_fit_review.score_policy',[('frame','pd.DataFrame')],
        [('offset','NDArray[np.float64]'),('lower','int'),('upper','int'),('ceiling','float')])
    add('validation_predict',base+'model_selection.materialized_prediction.predict',[('model','object'),('frame','pd.DataFrame')],[('values','NDArray[np.float64]')])
    add('validation_clip',base+'tabular.available_features.clip_truncate',
        [('values','NDArray[np.float64]'),('offset','NDArray[np.float64]'),('lower','int'),('upper','int')],[('result','NDArray[np.int64]')])
    edge('validation_predict','values','validation_clip','values')
    for name in ['offset','lower','upper']:edge('score_policy',name,'validation_clip',name)
    add('score',base+'model_selection.fit_review.absolute_error',[('observed','np.ndarray'),('predicted','np.ndarray')],[('score','float')])
    edge('validation_clip','result','score','predicted')
    add('accept',base+'model_selection.fit_review.accept',[('model','object'),('score','float'),('ceiling','float')],[('model','object')])
    edge('score','score','accept','score');edge('score_policy','ceiling','accept','ceiling')
    if residual:
        add('importance',base+'model_selection.fit_review.importance',[('model','object')],[('names','list'),('values','np.ndarray')])
        edge('accept','model','importance','model')
        add('normalize',base+'model_selection.fit_review.normalize',[('names','list'),('values','np.ndarray')],[('importance','pd.DataFrame')])
        edge('importance','names','normalize','names');edge('importance','values','normalize','values')
        add('residual_outputs',base+'domain_adapters.first_place_fit_review.residual_outputs',[('model','object'),('importance','pd.DataFrame')],
            [('model_v0','object'),('model_v2','object'),('feature_importance','pd.DataFrame')])
        edge('accept','model','residual_outputs','model');edge('normalize','importance','residual_outputs','importance')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',residual=residual,
        scope='Source single-candidate acceptance and optional residual model aliases/importance.'))


def build_nasa_first_training_graph(*, residual):
    graph=build_nasa_first_fit_graph(residual=residual)
    review=build_fit_review_graph(residual=residual)
    graph.nodes.extend(review.nodes);graph.edges.extend(review.edges)
    connected={(edge.target_id,edge.input_name) for edge in review.edges}
    for node in review.nodes:
        for port in node.inputs:
            if (node.node_id,port.name) in connected:continue
            source,output={'frame':('split','x_valid'),'observed':('split','y_valid'),'model':('fit','model')}[port.name]
            graph.edges.append(DependencyEdge(source_id=source,target_id=node.node_id,output_name=output,input_name=port.name,
                source_type=port.type_desc,target_type=port.type_desc))
    graph.metadata=dict(publication_status='draft',residual=residual,
        scope='Complete source single-fit input preparation, native training, score acceptance and residual alias/importance outputs.',
        exclusions=['21-fit population assembly','Raw feature extraction','Native state serialization'])
    return graph

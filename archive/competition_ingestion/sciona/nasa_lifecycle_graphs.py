"""Training and inference slices retaining explicit domain and numerical nodes."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.nasa_domain_graph import build_nasa_domain_graph,domain_contract

PREFIX='sciona.atoms.ml.domain_adapters.airport_state.'


def _node(identity,function,inputs,outputs):
    return AlgorithmicNode(node_id=identity,name=function.replace('_',' '),description='Explicit learned-state lifecycle boundary.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PREFIX+function,
        inputs=[IOSpec(name=name,type_desc=typ,constraints=contract) for name,typ,contract in inputs],
        outputs=[IOSpec(name=name,type_desc=typ,constraints=contract) for name,typ,contract in outputs])


def _edge(source,target,output,name,typ):
    return DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=name,source_type=typ,target_type=typ)


def _ancestors(graph,sinks):
    selected=set(sinks)
    while True:
        before=len(selected)
        selected.update(edge.source_id for edge in graph.edges if edge.target_id in selected)
        if len(selected)==before:break
    return [node for node in graph.nodes if node.node_id in selected],[edge for edge in graph.edges if edge.target_id in selected]


def build_nasa_training_graph():
    graph=build_nasa_domain_graph().model_copy(deep=True)
    handoff=next(node for node in graph.nodes if node.node_id=='domain_handoff')
    handoff.matched_primitive=PREFIX+'training_handoff'
    handoff.inputs=[port for port in handoff.inputs if port.name!='prediction_features']
    handoff.outputs=[port for port in handoff.outputs if port.name!='prediction_features']
    handoff.description='Training-only metadata check; observed training matrix is a shape probe, not future query evidence.'
    graph.edges=[edge for edge in graph.edges if not (edge.target_id=='domain_handoff' and edge.input_name=='prediction_features')]
    nodes,edges=_ancestors(graph,['offsets'])
    specifications=[('feature_names','list','domain_train_arrays','feature_names'),
        ('prediction_feature_name','str','domain_handoff','prediction_feature_name'),
        ('regressor','dict','lower_fit','state'),('classifier','dict','classifier_fit','state'),
        ('offsets','NDArray[np.float64]','offsets','offsets'),('threshold','float','domain_handoff','threshold'),
        ('vocabulary','tuple','domain_vocabulary','vocabulary'),('airport','str',None,None)]
    inputs=[]
    for name,typ,source,output in specifications:
        contract=('Exact native backend model, task, ordered feature schema and integrity digest; private runtime value.'
                  if name in ('regressor','classifier') else
                  'Finite signed float64 offset pair in minutes, ordered underestimation then overestimation.' if name=='offsets' else
                  domain_contract('numerical_handoff' if name in ('prediction_feature_name','threshold') else 'training_arrays',name))
        inputs.append((name,typ,contract))
        if source:edges.append(_edge(source,'learned_state',output,name,typ))
    nodes.append(_node('learned_state','pack_model_state',inputs,[('model_state','dict',
        'JSON-compatible complete native models, feature schema, vocabulary, airport, minute units, calibration and probability policy. Private runtime state; no query data or refitting is embedded.')]))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(graph.metadata,num_nodes=len(nodes),num_edges=len(edges),
        scope='Training-only corrected airport residual classifier; explicit complete learned-state output.',
        exclusions=['Whole-graph static table propagation','Multi-airport graph dispatch','Empirical competition performance']))


def build_nasa_inference_graph():
    graph=build_nasa_domain_graph()
    selected={'domain_query','domain_query_departures','domain_query_categories','domain_query_history',
        'domain_query_join','domain_query_arrays','query','query_predict','query_classifier_features',
        'query_probabilities','correct','final','domain_output'}
    outputs=[('airport','str'),('feature_names','list'),('vocabulary','tuple'),('regressor','dict'),
        ('classifier','dict'),('offsets','NDArray[np.float64]'),('threshold','float'),
        ('prediction_feature_name','str'),('classifier_feature_names','list')]
    state_node=_node('loaded_state','unpack_model_state',[('model_state','dict',
        'Complete decoded residual-classifier state with exact backend model tasks, feature order and minute-based airport adapter metadata.')],
        [(name,typ,'Validated learned state; preserve schema, category vocabulary, model task, calibration order and target scale without refitting.') for name,typ in outputs])
    nodes=[state_node]+[node.model_copy(deep=True) for node in graph.nodes if node.node_id in selected]
    mapping={('domain_vocabulary','vocabulary'):'vocabulary',('domain_train_arrays','feature_names'):'feature_names',
        ('training','feature_names'):'feature_names',('lower_fit','state'):'regressor',
        ('classifier_fit','state'):'classifier',('classifier_names','names'):'classifier_feature_names',
        ('offsets','offsets'):'offsets',('domain_handoff','threshold'):'threshold'}
    edges=[]
    for edge in graph.edges:
        if edge.target_id not in selected:continue
        if edge.source_id in selected:edges.append(edge.model_copy(deep=True))
        elif (edge.source_id,edge.output_name)==('domain_handoff','prediction_features'):
            edges.append(_edge('domain_query_arrays',edge.target_id,'features',edge.input_name,edge.target_type))
        else:
            key=(edge.source_id,edge.output_name)
            if key not in mapping:raise ValueError('Unmapped inference state dependency: '+str(key))
            edges.append(_edge('loaded_state',edge.target_id,mapping[key],edge.input_name,edge.target_type))
    for node in nodes[1:]:
        for port in node.inputs:
            if port.name=='airport':edges.append(_edge('loaded_state',node.node_id,'airport','airport','str'))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(graph.metadata,num_nodes=len(nodes),num_edges=len(edges),
        scope='Inference-only corrected airport residual classifier from complete saved state; no fitting operations.',
        exclusions=['Whole-graph static table propagation','Multi-airport graph dispatch','Empirical competition performance']))

"""All 21 source fits, selected global population, and explicit model-bank wiring."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.nasa_first_training_graph import build_nasa_first_training_graph


def build_nasa_first_population_training_graph():
    nodes,edges=[],[]
    def p(name,typ):return IOSpec(name=name,type_desc=typ,constraints='Explicit source-ordered runtime population, model or policy value; private for non-public inputs.')
    def add(identity,runtime,inputs,outputs):
        nodes.append(AlgorithmicNode(node_id=identity,name=identity,description='Source population training orchestration',concept_type='data_assembly',
            status=NodeStatus.ATOMIC,matched_primitive=runtime,inputs=[p(*port) for port in inputs],outputs=[p(*port) for port in outputs]))
    def edge(source,output,target,name):
        before=next(port for node in nodes if node.node_id==source for port in node.outputs if port.name==output)
        after=next(port for node in nodes if node.node_id==target for port in node.inputs if port.name==name)
        edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=name,
                                    source_type=before.type_desc,target_type=after.type_desc))
    add('populations','sciona.atoms.ml.domain_adapters.first_place_populations.training_tables',
        [('population_tables','dict'),('target_name','str')],[(f'table_{i}','pd.DataFrame') for i in range(10)]+
        [('tables','list'),('labels','list'),('columns','list'),('label_column','str'),('order_column','str')])
    global_ports=[('tables','list'),('labels','list'),('columns','list'),('label_column','str'),('order_column','str')]
    add('global_table','sciona.atoms.ml.model_selection.labeled_populations.concatenate',global_ports,[('table','pd.DataFrame')])
    for name,_ in global_ports:edge('populations',name,'global_table',name)
    model_sources=[]
    for population in range(11):
        for residual in ([True,False] if population<10 else [False]):
            prefix=('residual' if residual else 'direct')+str(population)+'/'
            subgraph=build_nasa_first_training_graph(residual=residual)
            for node in subgraph.nodes:
                nodes.append(node.model_copy(update={'node_id':prefix+node.node_id},deep=True))
            for connection in subgraph.edges:
                edges.append(connection.model_copy(update={'source_id':prefix+connection.source_id,'target_id':prefix+connection.target_id},deep=True))
            edge('populations' if population<10 else 'global_table',f'table_{population}' if population<10 else 'table',prefix+'inputs','table')
            model_sources.append((prefix+'residual_outputs' if residual else prefix+'accept','model_v0' if residual else 'model'))
    binding_inputs=[(name+str(i),'object') for i in range(10) for name in ['residual','direct']]+[('global_model','object')]
    bank_inputs=[('models','dict'),('population_bindings','dict'),('shared_bindings','dict')]
    add('bindings','sciona.atoms.ml.domain_adapters.first_place_populations.model_bindings',binding_inputs,bank_inputs)
    for (source,output),(name,_) in zip(model_sources,binding_inputs):edge(source,output,'bindings',name)
    add('bank','sciona.atoms.ml.model_selection.model_banks.assemble_bank',bank_inputs,[('bank','dict')])
    for name,_ in bank_inputs:edge('bindings',name,'bank',name)
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',independent_fit_nodes=21,
        residual_alias_pairs=10,local_model_slots=30,shared_global_models=1,
        scope='Complete source training topology from prepared population feature tables, including source validation and residual importance.',
        exclusions=['Raw feature preparation and training target attachment','Native state serialization','Empirical predictive quality']))

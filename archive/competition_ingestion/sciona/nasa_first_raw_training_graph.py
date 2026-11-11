"""Raw population preparation composed with all 21 native source fits."""
import sciona.atoms.ml.tabular.available_features as features
from sciona.available_feature_contracts import contracts
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.nasa_first_feature_graph import build_nasa_first_feature_graph
from sciona.nasa_first_population_training_graph import build_nasa_first_population_training_graph


def build_raw_training_preparation_graph():
    nodes,edges=[],[]
    def p(name,typ):return IOSpec(name=name,type_desc=typ,constraints='Explicit caller-owned training population values and fixed policy; preserve row alignment and declared units.')
    def add(identity,runtime,inputs,outputs):
        nodes.append(AlgorithmicNode(node_id=identity,name=identity,description='Raw training population preparation',concept_type='data_assembly',
            status=NodeStatus.ATOMIC,matched_primitive=runtime,inputs=inputs,outputs=outputs))
    def edge(source,output,target,name):
        before=next(port for node in nodes if node.node_id==source for port in node.outputs if port.name==output)
        after=next(port for node in nodes if node.node_id==target for port in node.inputs if port.name==name)
        edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=name,
                                    source_type=before.type_desc,target_type=after.type_desc))
    raw_ports=[('queries','pd.DataFrame'),('raw_options','dict'),('feature_columns','list'),
               ('numeric_fills','dict'),('categorical_fills','dict'),('integer_dtype','str')]
    base='sciona.atoms.ml.domain_adapters.first_place_raw_training.'
    add('raw_inputs',base+'raw_inputs',[p('raw_populations','dict')],
        [p(f'population{i}_{name}',typ) for i in range(10) for name,typ in raw_ports])
    for index in range(10):
        prefix=f'raw{index}/';subgraph=build_nasa_first_feature_graph()
        nodes.extend(node.model_copy(update={'node_id':prefix+node.node_id},deep=True) for node in subgraph.nodes)
        edges.extend(connection.model_copy(update={'source_id':prefix+connection.source_id,'target_id':prefix+connection.target_id},deep=True) for connection in subgraph.edges)
        for name in ['queries','raw_options']:edge('raw_inputs',f'population{index}_{name}',prefix+'prepare',name)
        inputs,outputs=contracts('prepare_features',features.prepare_features)
        add(prefix+'model_features','sciona.atoms.ml.tabular.available_features.prepare_features',
            [IOSpec(**port) for port in inputs],[IOSpec(**port) for port in outputs])
        edge(prefix+'assemble','result',prefix+'model_features','frame')
        for name in ['feature_columns','numeric_fills','categorical_fills','integer_dtype']:
            edge('raw_inputs',f'population{index}_{name}',prefix+'model_features',name)
        add(prefix+'training_table',base+'attach_target',[p('queries','pd.DataFrame'),p('frame','pd.DataFrame'),p('target_name','str')],[p('table','pd.DataFrame')])
        edge('raw_inputs',f'population{index}_queries',prefix+'training_table','queries')
        edge(prefix+'model_features','result',prefix+'training_table','frame')
    add('collect_tables',base+'collect_tables',[p(f'table{i}','pd.DataFrame') for i in range(10)],[p('population_tables','dict')])
    for index in range(10):edge(f'raw{index}/training_table','table','collect_tables',f'table{index}')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',
        scope='All ten raw feature populations, fixed model policies and positional training target attachment.'))


def build_nasa_first_raw_training_graph():
    graph=build_raw_training_preparation_graph()
    training=build_nasa_first_population_training_graph()
    graph.nodes.extend(training.nodes);graph.edges.extend(training.edges)
    graph.edges.append(DependencyEdge(source_id='collect_tables',target_id='populations',output_name='population_tables',
        input_name='population_tables',source_type='dict',target_type='dict'))
    graph.metadata=dict(publication_status='draft',independent_fit_nodes=21,residual_alias_pairs=10,
        scope='Raw populations through all feature operations, 21 fits, validation/importance and shared model bank.',
        exclusions=['Inference policy binding','Portable model-bank output encoding','Empirical predictive quality'])
    return graph

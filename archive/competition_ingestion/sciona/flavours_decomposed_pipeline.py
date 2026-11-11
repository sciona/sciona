"""Complete 115-fit training/prediction CDG, with evaluation kept external."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.flavours_mass_execution import build_mass_regression_graph
from sciona.flavours_classifier_execution import build_classifier_graph
from sciona.flavours_neural_execution import build_neural_graph


def build_pipeline_graph():
    nodes,edges=[],[];array='numpy.ndarray';base='sciona.atoms.ml.'
    def node(name,primitive,inputs,outputs,description):
        nodes.append(AlgorithmicNode(node_id=name,name=name,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=base+primitive,description=description,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,output,target,port,kind=array):
        edges.append(DependencyEdge(source_id=source,output_name=output,target_id=target,input_name=port,source_type=kind,target_type=kind))
    def compose(prefix,graph,roots):
        consumed={(e.target_id,e.input_name) for e in graph.edges}
        for original in graph.nodes:
            nodes.append(original.model_copy(deep=True,update={'node_id':prefix+original.node_id}))
            for port in original.inputs:
                if (original.node_id,port.name) not in consumed:
                    source,output=roots[port.name]
                    edge(source,output,prefix+original.node_id,port.name,port.type_desc)
        edges.extend(e.model_copy(deep=True,update={'source_id':prefix+e.source_id,'target_id':prefix+e.target_id}) for e in graph.edges)
    physical=['base','momentum','transverse_momentum','parent_transverse_momentum','flight_distance','lifetime']
    node('inputs','flavours_pipeline_wiring.training_prediction_inputs',
        [('training_inputs','list'),('labels',array),('mass',array),('query_inputs','list'),('excluded_column','int')],
        [(population+'_'+name,array) for population in ('training','query') for name in physical]
        +[('labels',array),('mass',array),('excluded_column','int')],
        'Preflight physical magnitudes and all source neural partitions before any model fitting.')
    for population in ('training','query'):
        kinematics=population+'_kinematics';prepared=population+'_prepared'
        node(kinematics,'flavours_feature_views.positive_longitudinal_kinematics',[(name,array) for name in physical[1:]],
            [(name,array) for name in ('mass_proxy','total_momentum','speed','longitudinal')],
            'Explicit positive-longitudinal physical proxy and auxiliary quantities.')
        for name in physical[1:]:edge('inputs',population+'_'+name,kinematics,name)
        node(prepared,'flavours_feature_views.assemble_source_feature_views',
            [(name,array) for name in ('base','mass_proxy','total_momentum','speed','longitudinal')]+[('excluded_column','int')],
            [('prepared','dict')],'Assemble source positional feature views with caller-selected exclusion.')
        edge('inputs',population+'_base',prepared,'base');edge('inputs','excluded_column',prepared,'excluded_column','int')
        for name in ('mass_proxy','total_momentum','speed','longitudinal'):edge(kinematics,name,prepared,name)
    node('regression_views','flavours_pipeline_wiring.regression_view_pair',[('training','dict'),('query','dict')],
        [('training',array),('query',array),('names','list')],'Route numerical regression views and positional model-state names.')
    for population in ('training','query'):edge(population+'_prepared','prepared','regression_views',population,'dict')
    compose('mass_',build_mass_regression_graph(),dict(features=('regression_views','training'),
        query_features=('regression_views','query'),feature_names=('regression_views','names'),targets=('inputs','mass')))
    for population,output in [('training','held_mean'),('query','query_mean')]:
        node(population+'_enriched','flavours_feature_views.enrich_corrected_mass_views',
            [('prepared','dict'),('corrected_mass',array)],[('views','dict')],'Append corrected mass, residual and source-stabilized ratio.')
        edge(population+'_prepared','prepared',population+'_enriched','prepared','dict')
        edge('mass_'+output,'predictions',population+'_enriched','corrected_mass')
    node('model_views','flavours_pipeline_wiring.classifier_neural_view_pair',[('training','dict'),('query','dict')],
        [('training_views','dict'),('query_views','dict'),('classifier_names','dict'),('neural_training',array),('neural_query',array),('neural_names','list')],
        'Expose corrected matrices separately for the neural and tree branches.')
    for population in ('training','query'):edge(population+'_enriched','views','model_views',population,'dict')
    compose('trees_',build_classifier_graph(),dict(training_views=('model_views','training_views'),
        query_views=('model_views','query_views'),feature_names=('model_views','classifier_names'),targets=('inputs','labels')))
    compose('neural_',build_neural_graph(),dict(features=('model_views','neural_training'),
        query_features=('model_views','neural_query'),feature_names=('model_views','neural_names'),targets=('inputs','labels')))
    roles=('first','second','third','fourth','fifth','sixth')
    for population in ('oof','query'):
        name=population+'_blend'
        node(name,'flavours_score_blend.six_model_power_blend',[(role,array) for role in roles],[('scores',array)],
            'Fixed source nonlinear blend; no calibration, tuning or diagnostic-driven modification.')
        for branch,role in enumerate(roles[:5],1):
            source=f'trees_xgb{branch}_'+('oof' if population=='oof' else 'full_prediction')
            edge(source,'predictions',name,role)
        edge('neural_'+('held_sum' if population=='oof' else 'query_sum'),'predictions',name,'sixth')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',target_tier=3,
        num_nodes=len(nodes),num_edges=len(edges),scope='Full physical-input training and prediction workflow: 35 mass, 30 classifier and 50 neural fits.',
        limitations=['Evaluation-control concatenation, splitting and diagnostics are external to this graph.',
            'Caller establishes source column meanings, consistent physical units and population disjointness.',
            'Upstream mass correction is not nested inside downstream classifier folds; OOF scores are diagnostic.',
            'No historical engine equivalence or predictive-quality claim.']))

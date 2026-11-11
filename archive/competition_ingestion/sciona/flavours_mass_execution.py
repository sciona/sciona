"""Decomposed 35-fit mass regression stage for the pending Flavours original."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_mass_regression_graph():
    nodes,edges=[],[]
    array='numpy.ndarray'
    base='sciona.atoms.ml.'
    def node(name,primitive,inputs,outputs,description):
        nodes.append(AlgorithmicNode(node_id=name,name=name,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=base+primitive,description=description,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,output,target,port,kind=array):
        edges.append(DependencyEdge(source_id=source,output_name=output,target_id=target,
            input_name=port,source_type=kind,target_type=kind))
    node('configuration','flavours_mass_configuration.mass_regression_inputs',
         [('features',array),('query_features',array)],
         [('training',array),('query',array),('query_selection',array),('n_splits','int')]
         +[(f'seed{i}','int') for i in range(7)]+[(f'fold{i}','int') for i in range(5)]
         +[('configuration','dict'),('boosting_rounds','int'),('held_coverage','int'),('query_coverage','int')],
         'Explicit numerical inputs and fixed source controls; no fitting or inferred population membership.')
    previous={}
    for repeat in range(7):
        for fold in range(5):
            prefix=f'r{repeat}_f{fold}'
            partition=prefix+'_partition';fit=prefix+'_fit'
            node(partition,'model_selection.configured_crossfit.shuffled_kfold_masks',
                 [('features',array),('n_splits','int'),('seed','int'),('fold_index','int')],
                 [('fit',array),('held',array)],'Target-independent shuffled partition retaining row order.')
            for output,port,kind in [('training','features',array),('n_splits','n_splits','int'),
                    (f'seed{repeat}','seed','int'),(f'fold{fold}','fold_index','int')]:
                edge('configuration',output,partition,port,kind)
            node(fit,'model_selection.configured_crossfit.fit_masked_configured_regression',
                 [('features',array),('targets',array),('feature_names','list'),('selection',array),
                  ('configuration','dict'),('boosting_rounds','int')],
                 [('state','dict'),('actual_rounds','int')],'Train only selected rows for all 2800 rounds.')
            edge('configuration','training',fit,'features');edge(partition,'fit',fit,'selection')
            for port,kind in [('configuration','dict'),('boosting_rounds','int')]:
                edge('configuration',port,fit,port,kind)
            for family in ('held','query'):
                prediction=prefix+'_'+family+'_prediction';contribution=prefix+'_'+family+'_contribution'
                node(prediction,'xgboost.model_io.predict_regression_model',
                     [('prediction_features',array),('feature_names','list'),('state','dict')],
                     [('predictions',array)],'Predict from the exact serialized native model.')
                edge('configuration','training' if family=='held' else 'query',prediction,'prediction_features')
                edge(fit,'state',prediction,'state','dict')
                node(contribution,'model_selection.prediction_accumulation.masked_prediction_contribution',
                     [('predictions',array),('selection',array)],[('total',array),('counts',array)],
                     'Promote native predictions to float64 and retain only the explicit selection.')
                edge(prediction,'predictions',contribution,'predictions')
                edge(partition if family=='held' else 'configuration',
                     'held' if family=='held' else 'query_selection',contribution,'selection')
                if family not in previous:
                    previous[family]=contribution
                else:
                    merge=prefix+'_'+family+'_merge'
                    node(merge,'model_selection.prediction_accumulation.merge_prediction_sums',
                         [('left',array),('left_counts',array),('right',array),('right_counts',array)],
                         [('total',array),('counts',array)],'Sequential source-order accumulation with overflow checks.')
                    for source,side in [(previous[family],'left'),(contribution,'right')]:
                        edge(source,'total',merge,side);edge(source,'counts',merge,side+'_counts')
                    previous[family]=merge
    for family in ('held','query'):
        node(family+'_mean','model_selection.prediction_accumulation.finalize_prediction_mean',
             [('total',array),('counts',array),('expected_coverage','int')],[('predictions',array)],
             'Require seven held-out or 35 query contributions per row before averaging.')
        edge(previous[family],'total',family+'_mean','total')
        edge(previous[family],'counts',family+'_mean','counts')
        edge('configuration',family+'_coverage',family+'_mean','expected_coverage','int')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',target_tier=3,
        num_nodes=len(nodes),num_edges=len(edges),scope='Complete repeated mass regression stage; not the complete Flavours workflow.',
        limitations=['Caller establishes feature semantics, units and population disjointness.',
                     'Corrected features are not nested within downstream classifier validation.',
                     'No historical engine equivalence or predictive quality claim.']))

"""Five explicit classifier branches: held-out fits and separate full refits."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_classifier_graph():
    nodes,edges=[],[];array='numpy.ndarray';base='sciona.atoms.ml.'
    def node(name,primitive,inputs,outputs,description):
        nodes.append(AlgorithmicNode(node_id=name,name=name,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=base+primitive,description=description,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,output,target,port,kind=array):
        edges.append(DependencyEdge(source_id=source,output_name=output,target_id=target,input_name=port,
            source_type=kind,target_type=kind))
    outputs=[]
    for view in ('base','restricted','proxy','corrected'):
        outputs.extend([(view+'_train',array),(view+'_query',array),(view+'_names','list')])
    outputs += [('n_splits','int'),('seed','int')]+[(f'fold{i}','int') for i in range(5)]+[('coverage','int')]
    for branch in range(1,6):outputs.extend([(f'config{branch}','dict'),(f'rounds{branch}','int')])
    node('configuration','flavours_classifier_configuration.classifier_view_inputs',
        [('training_views','dict'),('query_views','dict'),('feature_names','dict')],outputs,
        'Explicit aligned numerical views and fixed source settings, with no hidden fitting.')
    for fold in range(5):
        partition=f'fold{fold}'
        node(partition,'model_selection.configured_crossfit.shuffled_kfold_masks',
            [('features',array),('n_splits','int'),('seed','int'),('fold_index','int')],
            [('fit',array),('held',array)],'Shared target-independent shuffled partition.')
        for output,port,kind in [('base_train','features',array),('n_splits','n_splits','int'),
                ('seed','seed','int'),(f'fold{fold}','fold_index','int')]:
            edge('configuration',output,partition,port,kind)
    for branch,view in enumerate(('proxy','restricted','base','proxy','corrected'),1):
        previous=None
        for fold in list(range(5))+['full']:
            prefix=f'xgb{branch}_{fold}';fit=prefix+'_fit';prediction=prefix+'_prediction'
            full=fold=='full'
            inputs=[('features',array),('targets',array),('feature_names','list'),('configuration','dict'),('boosting_rounds','int')]
            if not full:inputs.append(('selection',array))
            node(fit,'xgboost.configured_model_io.fit_configured_binary_model' if full else
                'model_selection.configured_crossfit.fit_masked_configured_binary',inputs,
                [('state','dict'),('actual_rounds','int')],
                'Full-data refit for query prediction.' if full else 'Fit only selected rows for held-out prediction.')
            for output,port,kind in [(view+'_train','features',array),(view+'_names','feature_names','list'),
                    (f'config{branch}','configuration','dict'),(f'rounds{branch}','boosting_rounds','int')]:
                edge('configuration',output,fit,port,kind)
            if not full:edge(f'fold{fold}','fit',fit,'selection')
            node(prediction,'xgboost.model_io.predict_binary_model',
                [('prediction_features',array),('feature_names','list'),('state','dict')],[('predictions',array)],
                'Predict from the serialized full refit.' if full else 'Predict before explicit held-row selection.')
            edge('configuration',view+('_query' if full else '_train'),prediction,'prediction_features')
            edge('configuration',view+'_names',prediction,'feature_names','list')
            edge(fit,'state',prediction,'state','dict')
            if full:continue
            contribution=prefix+'_contribution'
            node(contribution,'model_selection.prediction_accumulation.masked_prediction_contribution',
                [('predictions',array),('selection',array)],[('total',array),('counts',array)],
                'Discard training-row predictions and retain held-out contribution counts.')
            edge(prediction,'predictions',contribution,'predictions');edge(f'fold{fold}','held',contribution,'selection')
            if previous is None:previous=contribution
            else:
                merge=prefix+'_merge'
                node(merge,'model_selection.prediction_accumulation.merge_prediction_sums',
                    [('left',array),('left_counts',array),('right',array),('right_counts',array)],
                    [('total',array),('counts',array)],'Merge disjoint held-out predictions in fold order.')
                for source,side in [(previous,'left'),(contribution,'right')]:
                    edge(source,'total',merge,side);edge(source,'counts',merge,side+'_counts')
                previous=merge
        final=f'xgb{branch}_oof'
        node(final,'model_selection.prediction_accumulation.finalize_prediction_mean',
            [('total',array),('counts',array),('expected_coverage','int')],[('predictions',array)],
            'Require exactly one held-out prediction for every training row.')
        edge(previous,'total',final,'total');edge(previous,'counts',final,'counts')
        edge('configuration','coverage',final,'expected_coverage','int')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',target_tier=3,
        num_nodes=len(nodes),num_edges=len(edges),scope='All five classifier branches and 30 fits; not the complete Flavours workflow.',
        limitations=['Caller supplies aligned views and int64 binary targets with both classes in every fit.',
            'OOF scores are diagnostic; upstream mass features are not nested in these folds.',
            'No historical engine equivalence or predictive-quality claim.']))

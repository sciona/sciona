"""Explicit 50-model neural graph with separate scaling and training masks."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_neural_graph():
    nodes,edges=[],[];array='numpy.ndarray';base='sciona.atoms.ml.'
    def node(name,primitive,inputs,outputs,description):
        nodes.append(AlgorithmicNode(node_id=name,name=name,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=base+primitive,description=description,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,output,target,port,kind=array):
        edges.append(DependencyEdge(source_id=source,output_name=output,target_id=target,input_name=port,source_type=kind,target_type=kind))
    def cfg(output,target,port,kind=array):edge('configuration',output,target,port,kind)
    node('configuration','flavours_neural_configuration.neural_ensemble_inputs',
        [('features',array),('targets',array),('query_features',array),('feature_names','list')],
        [('training',array),('targets',array),('query',array),('names','list'),('query_selection',array),('n_splits','int'),('seed','int')]
        +[(f'fold{i}','int') for i in range(5)]+[('inner_splits','int'),('inner_fold','int'),('held_divisor','float'),('query_divisor','float'),
        ('held_coverage','int'),('query_coverage','int')]+[(f'config{i}','dict') for i in range(10)],
        'Preflight every source partition before fitting; expose exact schedule and ordered numerical inputs.')
    previous={}
    for fold in range(5):
        prefix=f'f{fold}';partition=prefix+'_partition';inner=prefix+'_inner';scaler=prefix+'_scaler'
        node(partition,'model_selection.configured_crossfit.shuffled_kfold_masks',
            [('features',array),('n_splits','int'),('seed','int'),('fold_index','int')],[('fit',array),('held',array)],'Outer shuffled partition.')
        for output,port,kind in [('training','features',array),('n_splits','n_splits','int'),('seed','seed','int'),(f'fold{fold}','fold_index','int')]:cfg(output,partition,port,kind)
        node(inner,'model_selection.nested_stratified_masks.stratified_inner_masks',
            [('labels',array),('outer_selection',array),('n_splits','int'),('fold_index','int')],[('fit',array),('validation',array)],
            'Nonshuffled stratified inner fit and diagnostic validation within outer training only.')
        cfg('targets',inner,'labels');cfg('inner_splits',inner,'n_splits','int');cfg('inner_fold',inner,'fold_index','int')
        edge(partition,'fit',inner,'outer_selection')
        node(scaler,'model_selection.masked_standardization.fit_masked_standardization',
            [('features',array),('feature_names','list'),('selection',array)],[('state','dict')],'Fit scaler on complete outer training selection.')
        cfg('training',scaler,'features');cfg('names',scaler,'feature_names','list');edge(partition,'fit',scaler,'selection')
        for family in ('training','query'):
            transform=prefix+'_'+family
            node(transform,'model_selection.masked_standardization.apply_standardization',
                [('features',array),('feature_names','list'),('state','dict')],[('features',array)],'Apply exact fitted scaler without refitting.')
            cfg(family,transform,'features');cfg('names',transform,'feature_names','list');edge(scaler,'state',transform,'state','dict')
        for seed in range(10):
            identity=f'{prefix}_s{seed}';fit=identity+'_fit';loss=identity+'_validation'
            node(fit,'numpy_adagrad_binary.fit_masked_adagrad_binary',
                [('features',array),('targets',array),('feature_names','list'),('selection',array),('configuration','dict')],
                [('state','dict'),('epochs','int'),('updates','int')],'All 3000 epochs; optimizer sees only inner training rows.')
            edge(prefix+'_training','features',fit,'features');cfg('targets',fit,'targets');cfg('names',fit,'feature_names','list')
            edge(inner,'fit',fit,'selection');cfg(f'config{seed}',fit,'configuration','dict')
            node(loss,'neural_evaluation.masked_neural_loss',
                [('features',array),('targets',array),('feature_names','list'),('state','dict'),('selection',array)],
                [('loss','float')],'Diagnostic validation loss with no feedback to fitting.')
            edge(prefix+'_training','features',loss,'features');cfg('targets',loss,'targets');cfg('names',loss,'feature_names','list')
            edge(fit,'state',loss,'state','dict');edge(inner,'validation',loss,'selection')
            for family in ('held','query'):
                prediction=identity+'_'+family+'_prediction';contribution=identity+'_'+family+'_contribution'
                inputs=[('features',array),('feature_names','list'),('state','dict')]
                if family=='held':inputs.append(('selection',array))
                node(prediction,'neural_evaluation.masked_neural_predictions' if family=='held' else 'numpy_adagrad_binary.predict_adagrad_binary',
                    inputs,[('predictions',array)],'Predict held-out rows only.' if family=='held' else 'Predict query from the same trained state.')
                edge(prefix+('_training' if family=='held' else '_query'),'features',prediction,'features')
                cfg('names',prediction,'feature_names','list');edge(fit,'state',prediction,'state','dict')
                if family=='held':edge(partition,'held',prediction,'selection')
                node(contribution,'model_selection.scaled_prediction_contributions.divided_prediction_contribution',
                    [('predictions',array),('selection',array),('divisor','float')],[('total',array),('counts',array)],
                    'Divide before ordered summation to preserve source rounding.')
                edge(prediction,'predictions',contribution,'predictions');cfg(family+'_divisor',contribution,'divisor','float')
                edge(partition if family=='held' else 'configuration','held' if family=='held' else 'query_selection',contribution,'selection')
                if family not in previous:previous[family]=contribution
                else:
                    merge=identity+'_'+family+'_merge'
                    node(merge,'model_selection.prediction_accumulation.merge_prediction_sums',
                        [('left',array),('left_counts',array),('right',array),('right_counts',array)],[('total',array),('counts',array)],
                        'Sequential fold/seed order, with contribution counts retained.')
                    for source,side in [(previous[family],'left'),(contribution,'right')]:
                        edge(source,'total',merge,side);edge(source,'counts',merge,side+'_counts')
                    previous[family]=merge
    for family in ('held','query'):
        final=family+'_sum'
        node(final,'model_selection.scaled_prediction_contributions.finalize_prediction_sum',
            [('total',array),('counts',array),('expected_coverage','int')],[('predictions',array)],
            'Require exactly ten held or fifty query contributions; no second normalization.')
        edge(previous[family],'total',final,'total');edge(previous[family],'counts',final,'counts');cfg(family+'_coverage',final,'expected_coverage','int')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',target_tier=3,
        num_nodes=len(nodes),num_edges=len(edges),scope='Full fifty-model neural stage; not the complete Flavours workflow.',
        limitations=['Caller establishes feature semantics, units and population disjointness.',
            'Modern NumPy float64 reconstruction, not historical Lasagne parity or predictive-quality evidence.']))

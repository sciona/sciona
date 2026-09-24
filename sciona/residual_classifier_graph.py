"""Complete reusable numerical residual-classifier training and prediction graph.

Domain feature extraction and multi-population orchestration remain adapters.
"""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_residual_classifier_graph():
    nodes,edges=[],[]
    F='NDArray[np.float64]'; P='NDArray[np.float32]'; I='NDArray[np.int32]'; G='NDArray[np.int64]'; B='NDArray[np.bool_]'
    base='sciona.atoms.ml.'
    namespaces=dict(contract='model_selection.training_contracts.',population='model_selection.population_masks.',masked='model_selection.masked_operations.',model='xgboost.model_io.',numeric='calibration.prediction_features.',correction='calibration.float32_correction.')
    def contract(node,name,typ):
        if name in ('features','prediction_features'):
            return 'Nonempty float64 rows/features matrix. NaN uses backend missing-value behavior; infinities rejected. Preserve declared column order, meanings and units. Classifier matrices append exactly one prediction column; training and query row counts may differ.'
        if name in ('feature_names','names'):
            return 'Unique ordered nonempty backend-compatible names, one per feature column. Preserve this order through native model state; classifier schema appends the explicit prediction feature name.'
        if name=='prediction_feature_name':
            return 'Nonempty backend-compatible new feature name, absent from the base schema; identifies the appended prediction in target units.'
        if name=='groups':
            return 'Int64 group identifiers aligned to training rows; at least two distinct groups; dimensionless. Identity encoding must preserve group membership.'
        if name in ('selection','train','held','left','right'):
            return 'Boolean vector aligned to original training rows; dimensionless. Split masks exhaustively partition rows by group. Retained mask intersects held-out membership with inclusive residual eligibility. Fit selections must be nonempty; selected classifier rows need both classes.'
        if name=='targets' and typ==G or name=='labels':
            return 'Int64 strict-underestimation labels aligned to training rows: 1 when rounded prediction is below observation, otherwise 0. Selected classifier population must contain both classes.'
        if name in ('targets','observed'):
            return 'Finite float64 observations aligned to original training rows in caller-declared target units, shared with predictions, offsets and maximum_error.'
        if name=='train_fraction':
            return 'Finite float fraction strictly between zero and one, applied to unique groups. Must leave nonempty training and complementary held-out groups.'
        if name=='seed':
            return 'Explicit unsigned 32-bit integer random seed for the group split; dimensionless.'
        if name=='maximum_error':
            return 'Finite nonnegative float threshold in target units. Retain abs(observed-rounded_internal_prediction) <= maximum_error; equality included.'
        if name=='threshold':
            return 'Finite float probability threshold in [0,1], dimensionless. Both strict calibration groups must be populated inside the held-out mask; equality is excluded from offset estimation and unchanged during correction.'
        if name=='state':
            return 'JSON-compatible native UBJ model state with exact task, backend version, ordered feature names and payload integrity digest. Private runtime material for non-public training inputs; one CPU thread per model operation.'
        if name=='offsets':
            return 'Finite float64 pair [median(observed-predicted | q>threshold), median(predicted-observed | q<threshold)] estimated only on held-out rows; signed target units. Application explicitly narrows offsets to float32 and rejects overflow.'
        if name in ('probabilities','calibration_probabilities') or node=='probability_widen':
            return 'Finite row-aligned probability of strict underestimation in [0,1], dimensionless. Float32 model probabilities widen exactly for calibration.'
        return 'Row-aligned numeric prediction in target units with the declared dtype. Training rounds float32 predictions to nearest-even int32 then widens exactly; inference classifier inputs remain unrounded. Final int32 conversion rejects overflow; no clipping, scaling or imputation.'
    def add(node_id,namespace,function,inputs,outputs):
        ports=[]
        for name,(typ,source,output) in inputs.items():
            ports.append(IOSpec(name=name,type_desc=typ,constraints=contract(node_id,name,typ)))
            if source is not None:
                edges.append(DependencyEdge(source_id=source,target_id=node_id,output_name=output,input_name=name,source_type=typ,target_type=typ))
        nodes.append(AlgorithmicNode(node_id=node_id,name=function.replace('_',' '),description='Reusable residual-classifier computation',concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=base+namespaces[namespace]+function,inputs=ports,outputs=[IOSpec(name=name,type_desc=typ,constraints=contract(node_id,name,typ)) for name,typ in outputs.items()]))
    def root(typ): return typ,None,None
    def edge(typ,node,out): return typ,node,out
    feature=edge(F,'training','features'); target=edge(F,'training','targets'); names=edge('list','training','feature_names')
    add('training','contract','training_inputs',dict(features=root(F),targets=root(F),groups=root(G),feature_names=root('list')),dict(features=F,targets=F,groups=G,feature_names='list'))
    add('query','contract','prediction_inputs',dict(prediction_features=root(F),feature_names=names),dict(features=F))
    add('split','population','grouped_holdout_masks',dict(groups=edge(G,'training','groups'),train_fraction=root('float'),seed=root('int')),dict(train=B,held=B))
    add('internal_fit','masked','fit_masked_regression',dict(features=feature,targets=target,feature_names=names,selection=edge(B,'split','train')),dict(state='dict'))
    add('internal_predict','model','predict_regression_model',dict(prediction_features=feature,feature_names=names,state=edge('dict','internal_fit','state')),dict(predictions=P))
    add('internal_round','numeric','round_to_int32',dict(values=edge(P,'internal_predict','predictions')),dict(values=I))
    add('internal_widen','numeric','widen_prediction_values',dict(values=edge(I,'internal_round','values')),dict(values=F))
    add('residual','population','residual_threshold_mask',dict(observed=target,predictions=edge(F,'internal_widen','values'),maximum_error=root('float')),dict(selection=B))
    add('retained','masked','intersect_masks',dict(left=edge(B,'split','held'),right=edge(B,'residual','selection')),dict(selection=B))
    add('lower_fit','masked','fit_masked_regression',dict(features=feature,targets=target,feature_names=names,selection=edge(B,'retained','selection')),dict(state='dict'))
    add('lower_training_predict','model','predict_regression_model',dict(prediction_features=feature,feature_names=names,state=edge('dict','lower_fit','state')),dict(predictions=P))
    add('lower_round','numeric','round_to_int32',dict(values=edge(P,'lower_training_predict','predictions')),dict(values=I))
    add('labels','numeric','underestimation_labels',dict(observed=target,predictions=edge(I,'lower_round','values')),dict(labels=G))
    add('classifier_features','numeric','append_prediction_feature',dict(features=feature,predictions=edge(I,'lower_round','values')),dict(features=F))
    add('classifier_names','contract','append_feature_name',dict(feature_names=names,prediction_feature_name=root('str')),dict(names='list'))
    cname=edge('list','classifier_names','names'); cfeatures=edge(F,'classifier_features','features')
    add('classifier_fit','masked','fit_masked_binary',dict(features=cfeatures,targets=edge(G,'labels','labels'),feature_names=cname,selection=edge(B,'split','held')),dict(state='dict'))
    add('calibration_probabilities','model','predict_binary_model',dict(prediction_features=cfeatures,feature_names=cname,state=edge('dict','classifier_fit','state')),dict(probabilities=P))
    add('probability_widen','numeric','widen_prediction_values',dict(values=edge(P,'calibration_probabilities','probabilities')),dict(values=F))
    add('lower_widen','numeric','widen_prediction_values',dict(values=edge(I,'lower_round','values')),dict(values=F))
    add('offsets','masked','estimate_masked_offsets',dict(observed=target,calibration_predictions=edge(F,'lower_widen','values'),calibration_probabilities=edge(F,'probability_widen','values'),threshold=root('float'),selection=edge(B,'split','held')),dict(offsets=F))
    add('query_predict','model','predict_regression_model',dict(prediction_features=edge(F,'query','features'),feature_names=names,state=edge('dict','lower_fit','state')),dict(predictions=P))
    add('query_classifier_features','numeric','append_prediction_feature',dict(features=edge(F,'query','features'),predictions=edge(P,'query_predict','predictions')),dict(features=F))
    add('query_probabilities','model','predict_binary_model',dict(prediction_features=edge(F,'query_classifier_features','features'),feature_names=cname,state=edge('dict','classifier_fit','state')),dict(probabilities=P))
    add('correct','correction','apply_float32_offsets',dict(predictions=edge(P,'query_predict','predictions'),probabilities=edge(P,'query_probabilities','probabilities'),offsets=edge(F,'offsets','offsets'),threshold=root('float')),dict(predictions=P))
    add('final','numeric','round_to_int32',dict(values=edge(P,'correct','predictions')),dict(predictions=I))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(artifact_source='reusable_computation_reconstruction',publication_status='draft',num_nodes=len(nodes),num_edges=len(edges),source_version_ids=['7d6f7b78-1dff-5757-9326-4edad55aaa68'],source_content_hashes=['5a9856b1f6a4db73e7ce12809248de193b63ad06401a6e73b8322744cb7c44bd'],scope='Complete grouped two-stage regression, residual classifier calibration and corrected prediction on supplied feature matrices.',exclusions=['Domain feature adapters','Complete original NASA workflows','Empirical predictive quality','Historical backend identity']))

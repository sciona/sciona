"""Reconstructed one-step prediction branch; classifier execution remains upstream."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_one_step_prediction_branch():
    """Convert class-labelled predictions, then enforce adjacent-pair antisymmetry.

    This branch consumes trained-classifier probabilities, not raw observations
    or features. It is a component of competition reconstruction, not a complete
    solution or catalog publication decision.
    """
    array='numpy.ndarray'
    return CDGExport(nodes=[
        AlgorithmicNode(node_id='class_score',name='Convert class probabilities',description='Compute positive minus negative class probability using explicit labels.',concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.causal_inference.estimators.probabilities.causal_direction_probability_score',
            inputs=[IOSpec(name='probabilities',type_desc=array,constraints='finite normalized probabilities, shape (2n,3), adjacent rows are reversed observation pairs'),IOSpec(name='class_labels',type_desc=array,constraints='integer labels -1,0,1 in column order')],
            outputs=[IOSpec(name='scores',type_desc=array)]),
        AlgorithmicNode(node_id='paired_score',name='Symmetrize directional scores',description='Project scores onto adjacent-pair antisymmetry.',concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.causal_inference.estimators.atoms.symmetrized_prediction_fusion',
            inputs=[IOSpec(name='predictions',type_desc=array)],outputs=[IOSpec(name='causal_scores',type_desc=array)]),
    ],edges=[DependencyEdge(source_id='class_score',target_id='paired_score',output_name='scores',input_name='predictions',source_type=array,target_type=array)],metadata={
        'artifact_source':'competition_execution_reconstruction','publication_status':'draft',
        'scope':'One-step classifier probability conversion and adjacent-pair symmetrization; trained classifier and feature extraction remain upstream.',
        'num_nodes':2,'num_edges':1,
    })


def build_three_system_prediction_graph():
    """Reconstruct score processing with a separate port for every producer.

    Boundary inputs are actual classifier outputs. The wider competition graph
    still needs trained classifiers and feature extraction upstream.
    """
    graph=build_one_step_prediction_branch()
    array='numpy.ndarray'
    prefix='sciona.atoms.causal_inference.estimators.'
    def node(identity,primitive,inputs,output):
        return AlgorithmicNode(node_id=identity,name=identity.replace('_',' '),description='Combine explicit classifier scores; classifier fitting remains upstream.',concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=prefix+primitive,
            inputs=[IOSpec(name=name,type_desc=array) for name in inputs],outputs=[IOSpec(name=output,type_desc=array)])
    graph.nodes.extend([
        node('independence_direction','atoms.two_stage_independence_direction',['independence_scores','direction_scores'],'id_scores'),
        node('left_right','atoms.left_right_decomposed_prediction',['left_scores','right_scores'],'lr_scores'),
        node('ensemble','ensemble_ports.three_system_causal_ensemble',['independence_direction','left_right','one_step','weights'],'causal_scores'),
    ])
    for source,output,input_name in [('independence_direction','id_scores','independence_direction'),('left_right','lr_scores','left_right'),('paired_score','causal_scores','one_step')]:
        graph.edges.append(DependencyEdge(source_id=source,target_id='ensemble',output_name=output,input_name=input_name,source_type=array,target_type=array))
    graph.metadata.update(scope='Three-system score processing from classifier outputs; feature extraction and classifier training/execution remain upstream.',num_nodes=5,num_edges=4)
    return graph


def build_trained_one_step_prediction_graph():
    """Run a supplied trained classifier before the reconstructed one-step branch."""
    graph=build_one_step_prediction_branch()
    array='numpy.ndarray'
    graph.nodes.insert(0,AlgorithmicNode(node_id='classifier',name='Execute trained classifier',description='Predict labelled probabilities from a fitted gradient boosting model and its feature matrix.',concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.sklearn.ensemble.gradient_boosting.prediction.gradient_boosting_class_probabilities',
        inputs=[IOSpec(name='estimator',type_desc='object',constraints='fitted in-memory GradientBoostingClassifier with integer causal labels -1,0,1'),IOSpec(name='X',type_desc=array,constraints='finite feature matrix, rows alternate observation orientations')],
        outputs=[IOSpec(name='probabilities',type_desc=array),IOSpec(name='class_labels',type_desc=array)]))
    for name in ['probabilities','class_labels']:
        graph.edges.append(DependencyEdge(source_id='classifier',target_id='class_score',output_name=name,input_name=name,source_type=array,target_type=array))
    graph.metadata.update(scope='Trained one-step classifier execution and score processing; feature extraction and model training remain upstream.',num_nodes=3,num_edges=3)
    return graph


def build_trained_three_system_prediction_graph():
    """Connect actual five-model prediction to all three score branches."""
    graph=build_three_system_prediction_graph()
    array='numpy.ndarray'
    outputs=['probabilities','class_labels','independence_scores','direction_scores','left_scores','right_scores']
    graph.nodes.insert(0,AlgorithmicNode(node_id='classifiers',name='Execute five trained classifiers',description='Predict aligned class-labelled probabilities for one-step, dependence, direction, left and right models.',concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_predictions.causal_classifier_predictions',
        inputs=[IOSpec(name='X',type_desc=array)]+[IOSpec(name=name,type_desc='object',constraints='fitted GradientBoostingClassifier with role-specific integer labels') for name in ['one_step_model','independence_model','direction_model','left_model','right_model']],
        outputs=[IOSpec(name=name,type_desc=array) for name in outputs]))
    for target,names in [('class_score',['probabilities','class_labels']),('independence_direction',['independence_scores','direction_scores']),('left_right',['left_scores','right_scores'])]:
        for name in names:
            graph.edges.append(DependencyEdge(source_id='classifiers',target_id=target,output_name=name,input_name=name,source_type=array,target_type=array))
    graph.metadata.update(scope='Five trained classifier predictions and three-system score processing; feature extraction and model training remain upstream.',num_nodes=6,num_edges=10)
    return graph


def build_causal_training_prediction_graph():
    """Train role-specific models and execute the full three-system score graph."""
    graph=build_trained_three_system_prediction_graph()
    names=['one_step_model','independence_model','direction_model','left_model','right_model']
    graph.nodes.insert(0,AlgorithmicNode(node_id='training',name='Train causal classifiers',description='Fit source-defined classifier roles from paired feature rows and labels; extraction stays upstream.',concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_training.train_causal_classifiers',
        inputs=[IOSpec(name='training_features',type_desc='numpy.ndarray'),IOSpec(name='training_labels',type_desc='numpy.ndarray')]+[IOSpec(name=name,type_desc=kind,required=False) for name,kind in [('n_estimators','int'),('max_depth','int'),('min_samples_split','int'),('learning_rate','float'),('random_state','int')]],
        outputs=[IOSpec(name=name,type_desc='object') for name in names]))
    for name in names:
        graph.edges.append(DependencyEdge(source_id='training',target_id='classifiers',output_name=name,input_name=name,source_type='object',target_type='object'))
    graph.metadata.update(scope='Training and prediction on pre-extracted paired features; original observation-to-feature extraction remains upstream.',num_nodes=7,num_edges=15)
    return graph


def build_raw_pair_causal_training_prediction_graph():
    """Draft observation-to-score reconstruction; historical feature parity pending."""
    graph=build_causal_training_prediction_graph()
    array='numpy.ndarray'
    graph.nodes.insert(0,AlgorithmicNode(node_id='features',name='Extract paired causal features',description='Extract the ordered feature reconstruction separately for training and prediction pairs, with explicit types and reversed labels.',concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.causal_inference.feature_primitives.selected_features.prepare_causal_feature_matrices',
        inputs=[IOSpec(name=name,type_desc='list') for name in ['training_pairs','prediction_pairs','training_types','prediction_types']]+[IOSpec(name='pair_labels',type_desc=array)],
        outputs=[IOSpec(name=name,type_desc=array) for name in ['training_features','training_labels','X']]))
    for target,name in [('training','training_features'),('training','training_labels'),('classifiers','X')]:
        graph.edges.append(DependencyEdge(source_id='features',target_id=target,output_name=name,input_name=name,source_type=array,target_type=array))
    graph.metadata.update(scope='Raw-pair feature reconstruction, training and prediction; historical feature parity and publication review pending.',num_nodes=8,num_edges=18)
    return graph

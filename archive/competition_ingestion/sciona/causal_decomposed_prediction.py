"""Five explicit training and prediction branches; feature decomposition pending."""
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.causal_decomposed_training import ROLES,build_explicit_causal_training_graph


def build_explicit_causal_prediction_graph():
    graph=build_explicit_causal_training_graph()
    graph.nodes=[n for n in graph.nodes if n.node_id!='classifiers']
    graph.edges=[e for e in graph.edges if e.source_id!='classifiers' and e.target_id!='classifiers']
    array='numpy.ndarray'
    prefix='sciona.atoms.ml.sklearn.ensemble.gradient_boosting.'
    def node(identity,primitive,inputs,outputs,description):
        graph.nodes.append(AlgorithmicNode(node_id=identity,name=identity.replace('_',' '),
            description=description,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=primitive,inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,target,output,input_name,kind):
        graph.edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,
            input_name=input_name,source_type=kind,target_type=kind))
    ports=[(role+'_model','object') for role in ROLES]
    node('model_roles',prefix+'causal_model_roles.validate_causal_model_roles',ports,ports,
        'Validate source classifier role labels without fitting or predicting.')
    for role in ROLES:
        predict='predict_'+role
        edge('fit_'+role,'model_roles','model',role+'_model','object')
        node(predict,prefix+'prediction.gradient_boosting_class_probabilities',
             [('estimator','object'),('X',array)],[('probabilities',array),('class_labels',array)],
             'Predict class-labelled probabilities with one supplied classifier.')
        edge('model_roles',predict,role+'_model','estimator','object')
        edge('features',predict,'X','X',array)
        if role=='one_step':
            for name in ['probabilities','class_labels']:edge(predict,'class_score',name,name,array)
        else:
            selected='positive_'+role
            node(selected,'sciona.atoms.causal_inference.estimators.probabilities.binary_positive_class_probability',
                 [('probabilities',array),('class_labels',array)],[('scores',array)],
                 'Select the probability of label +1 using explicit class labels.')
            for name in ['probabilities','class_labels']:edge(predict,selected,name,name,array)
            target='independence_direction' if role in {'independence','direction'} else 'left_right'
            edge(selected,target,'scores',role+'_scores',array)
    graph.metadata.update(scope='Five explicit fits and predictions with recipe role validation; raw feature decomposition pending.',
        num_nodes=len(graph.nodes),num_edges=len(graph.edges),publication_status='draft')
    return graph

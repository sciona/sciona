"""Explicit five-fit causal recipe; feature/prediction decomposition is pending."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.causal_prediction_execution import build_raw_pair_causal_training_prediction_graph

ROLES = ('one_step','independence','direction','left','right')


def build_explicit_causal_training_graph():
    graph = build_raw_pair_causal_training_prediction_graph()
    training = next(node for node in graph.nodes if node.node_id == 'training')
    graph.nodes = [node for node in graph.nodes if node.node_id != 'training']
    graph.edges = [edge for edge in graph.edges if edge.source_id != 'training' and edge.target_id != 'training']
    prefix = 'sciona.atoms.ml.sklearn.ensemble.gradient_boosting.'
    array = 'numpy.ndarray'
    graph.nodes.append(AlgorithmicNode(
        node_id='partitions',name='Prepare source training roles',
        description='Partition observations and construct explicit role labels and sample weights without fitting.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive=prefix+'causal_partitions.prepare_causal_training_partitions',
        inputs=[IOSpec(name=n,type_desc=array) for n in ['training_features','training_labels']],
        outputs=[IOSpec(name=role+'_'+kind,type_desc='object' if kind=='weight' else array)
                 for role in ROLES for kind in ['features','labels','weight']],
    ))
    def connect(source,target,output,input_name,kind):
        graph.edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,
            input_name=input_name,source_type=kind,target_type=kind))
    for name in ['training_features','training_labels']:
        connect('features','partitions',name,name,array)
    for role in ROLES:
        identity = 'fit_'+role
        graph.nodes.append(AlgorithmicNode(
            node_id=identity,name='Fit '+role.replace('_',' ')+' classifier',
            description='Fit one classifier using explicit labels, weights and boosting settings.',
            concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=prefix+'weighted_fit.fit_weighted_gradient_boosting',
            inputs=[IOSpec(name='X',type_desc=array),IOSpec(name='labels',type_desc=array),
                    IOSpec(name='sample_weight',type_desc='object')]+
                   [item.model_copy(deep=True) for item in training.inputs if item.name not in {'training_features','training_labels'}],
            outputs=[IOSpec(name='model',type_desc='object')],
        ))
        for output,input_name,kind in [('features','X',array),('labels','labels',array),('weight','sample_weight','object')]:
            connect('partitions',identity,role+'_'+output,input_name,kind)
        connect(identity,'classifiers','model',role+'_model','object')
    graph.metadata.update(
        scope='Five explicit generic classifier fits with source partition adapter; full feature and prediction decomposition pending.',
        publication_status='draft',num_nodes=len(graph.nodes),num_edges=len(graph.edges),
    )
    return graph

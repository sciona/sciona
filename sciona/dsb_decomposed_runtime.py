"""Expose shared training runtime restoration before the stateful epoch loop."""
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.dsb_decomposed_inference import build_dsb_inference_steps_graph


def build_dsb_runtime_graph():
    graph=build_dsb_inference_steps_graph()
    training=next(n for n in graph.nodes if n.node_id=='training')
    prefix='sciona.atoms.dl.detection.dsb_training_runtime.'
    training.matched_primitive=prefix+'train_shared_runtime'
    training.description='Run source epochs with one explicit persistent shared model/optimizer runtime.'
    training.inputs=[p for p in training.inputs if p.name not in ('initial_classifier_checkpoint','start_epoch')]
    training.inputs.insert(0,IOSpec(name='runtime',type_desc='object'))
    graph.nodes.append(AlgorithmicNode(node_id='training_runtime',name='Restore shared training runtime',
        description='Restore shared networks and fresh optimizers once before the consecutive epoch loop.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=prefix+'restore_shared_training_runtime',
        inputs=[IOSpec(name='initial_classifier_checkpoint',type_desc='object'),
                IOSpec(name='topk',type_desc='int',required=False),IOSpec(name='start_epoch',type_desc='int',required=False)],
        outputs=[IOSpec(name='runtime',type_desc='object')]))
    for edge in graph.edges:
        if edge.target_id=='training' and edge.input_name=='initial_classifier_checkpoint':edge.target_id='training_runtime'
    graph.edges.append(DependencyEdge(source_id='training_runtime',target_id='training',output_name='runtime',input_name='runtime',
        source_type='object',target_type='object'))
    graph.metadata.update(scope='Explicit shared training-runtime restoration, reusable sampling/tiling and separate inference stages.',
        num_nodes=len(graph.nodes),num_edges=len(graph.edges),publication_status='draft',
        remaining_training_obligations=['Separate preparation, differentiable loss, validation and checkpoint operations.',
            'Qualify exact provider interfaces/dependencies and stored publication behavior.'])
    return graph

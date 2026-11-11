"""Draft DSB graph reconstruction; training obligations remain to be connected."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_dsb_inference_graph():
    prefix='sciona.atoms.dl.detection.dsb_execution.'
    def node(identity,primitive,inputs,outputs,optional=()):
        return AlgorithmicNode(node_id=identity,name=identity.replace('_',' '),
            description=primitive,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=prefix+primitive,
            inputs=[IOSpec(name=n,type_desc=t,required=n not in optional) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs])
    options=[('topk','int'),('crop_size','int'),('tile_side','int'),('margin','int'),('tile_batch_size','int')]
    nodes=[
        node('models','restore_dsb_models',[('detector_checkpoint','object'),('classifier_checkpoint','object'),('topk','int')],
             [('detector','object'),('classifier','object')],['topk']),
        node('preprocess','dsb_preprocess',[('volume','numpy.ndarray'),('spacing','object'),('requested_spacing','object')],
             [('processed_volume','numpy.ndarray')],['requested_spacing']),
        node('prediction','dsb_predict',[('processed_volume','numpy.ndarray'),('detector','object'),('classifier','object'),*options],
             [('case_probability','numpy.ndarray'),('proposal_probabilities','numpy.ndarray')],[n for n,_ in options]),
    ]
    edges=[DependencyEdge(source_id=source,target_id='prediction',output_name=name,input_name=name,
                          source_type=kind,target_type=kind)
           for source,name,kind in [('models','detector','object'),('models','classifier','object'),
                                     ('preprocess','processed_volume','numpy.ndarray')]]
    return CDGExport(nodes=nodes,edges=edges,metadata={
        'artifact_source':'competition_execution_reconstruction','publication_status':'draft',
        'source_version_id':'6625a41e-b5a7-5ee2-86e5-1d2c60d6d185',
        'source_commit':'0ac3eb9f383bf0127c587e0502c59ac84d9ba6a2',
        'scope':'Inference component from runtime arrays and explicit model checkpoints; full competition training remains unfinished.',
        'remaining_training_obligations':['size oversampling','training proposal sampling',
            'anchor assignment','OHEM loss','miss penalty','alternating training lifecycle'],
        'num_nodes':3,'num_edges':3,
    })


def build_dsb_training_inference_graph():
    """One warm-restarted epoch from preprocessed training arrays, then inference.

    Validation loaders and training annotation/proposal preparation remain explicit
    runtime inputs. Repeated nodes would restart optimizers, not continue training.
    """
    graph=build_dsb_inference_graph()
    prefix='sciona.atoms.dl.detection.dsb_training_execution.'
    graph.nodes.extend([
        AlgorithmicNode(node_id='training_batches',name='Prepare training batches',
            description='Execute oversampling, proposal sampling, augmentation and anchor assignment.',
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=prefix+'dsb_training_batches',
            inputs=[IOSpec(name=n,type_desc='object') for n in ['training_arrays','sampling_options']],
            outputs=[IOSpec(name='training_factory',type_desc='object')]),
        AlgorithmicNode(node_id='training',name='Train one warm-restarted epoch',
            description='Restore shared models and run source losses, validation and checkpoint emission.',
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=prefix+'dsb_train_epoch',
            inputs=[IOSpec(name=n,type_desc=t,required=required) for n,t,required in [
                ('initial_classifier_checkpoint','object',True),('training_factory','object',True),
                ('validation_factory','object',True),('epoch','int',True),('start_epoch','int',False),
                ('topk','int',False),('classifier_variant','int',False),('freeze_batchnorm','bool',False)]],
            outputs=[IOSpec(name='classifier_checkpoint',type_desc='object'),IOSpec(name='metrics',type_desc='object')]),
    ])
    for source,target,name in [('training_batches','training','training_factory'),('training','models','classifier_checkpoint')]:
        graph.edges.append(DependencyEdge(source_id=source,target_id=target,output_name=name,input_name=name,
                                         source_type='object',target_type='object'))
    graph.metadata.update(scope='One warm-restarted epoch from preprocessed training arrays and supplied validation batches, followed by array inference.',
        remaining_training_obligations=['raw training preprocessing and annotation/proposal preparation',
            'validation batch construction','multi-epoch optimizer continuity'],num_nodes=5,num_edges=5)
    return graph


def build_dsb_training_range_inference_graph():
    """Consecutive epochs retain optimizer state before the inference checkpoint edge."""
    graph=build_dsb_training_inference_graph()
    node=next(node for node in graph.nodes if node.node_id=='training')
    node.matched_primitive='sciona.atoms.dl.detection.dsb_training_execution.dsb_train_range'
    node.name='Train consecutive epochs'
    node.description='Restore once and preserve both optimizer states across the source epoch schedule.'
    next(port for port in node.inputs if port.name=='epoch').name='end_epoch'
    node.inputs.append(IOSpec(name='prepare_epoch',type_desc='object',required=False))
    graph.metadata.update(scope='Consecutive source epochs from preprocessed training arrays and supplied validation batches, followed by array inference.',
        remaining_training_obligations=['raw training preprocessing and annotation/proposal preparation',
            'validation batch construction'])
    return graph


def build_dsb_array_training_inference_graph():
    """Build validation batches within the persistent training/inference graph."""
    graph=build_dsb_training_range_inference_graph()
    graph.nodes.append(AlgorithmicNode(node_id='validation_batches',name='Prepare validation batches',
        description='Construct detector and both classifier validation loaders with source proposal annotation.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.dl.detection.dsb_training_execution.dsb_validation_batches',
        inputs=[IOSpec(name=n,type_desc='object') for n in ['validation_arrays','validation_options']],
        outputs=[IOSpec(name='validation_factory',type_desc='object')]))
    graph.edges.append(DependencyEdge(source_id='validation_batches',target_id='training',
        output_name='validation_factory',input_name='validation_factory',source_type='object',target_type='object'))
    graph.metadata.update(scope='Training and validation batch construction from preprocessed runtime arrays, consecutive epochs and inference.',
        remaining_training_obligations=['raw training preprocessing and training annotation/proposal preparation'],
        num_nodes=len(graph.nodes),num_edges=len(graph.edges))
    return graph


def build_dsb_detector_initialized_graph():
    """Include source detector-to-classifier initialization before training."""
    graph=build_dsb_array_training_inference_graph()
    graph.nodes.append(AlgorithmicNode(node_id='initialize_classifier',name='Initialize classifier from detector',
        description='Transfer detector weights to the shared network and initialize the classifier head.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.dl.detection.dsb_training_execution.dsb_adapt_detector',
        inputs=[IOSpec(name='detector_checkpoint',type_desc='object'),IOSpec(name='topk',type_desc='int',required=False)],
        outputs=[IOSpec(name='initial_classifier_checkpoint',type_desc='object')]))
    graph.edges.append(DependencyEdge(source_id='initialize_classifier',target_id='training',
        output_name='initial_classifier_checkpoint',input_name='initial_classifier_checkpoint',
        source_type='object',target_type='object'))
    graph.metadata.update(scope='Detector checkpoint transfer, array batch construction, consecutive training epochs and inference.',
                          num_nodes=len(graph.nodes),num_edges=len(graph.edges))
    return graph


def build_dsb_raw_training_graph():
    """Include raw preparation; detector states and proposal arrays remain runtime inputs."""
    graph=build_dsb_detector_initialized_graph()
    graph.nodes.append(AlgorithmicNode(node_id='raw_preparation',name='Prepare raw training and validation inputs',
        description='Apply source-specific volume and annotation preparation and label training proposals.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.dl.detection.dsb_training_execution.dsb_raw_collections',
        inputs=[IOSpec(name=n,type_desc='object') for n in ['raw_collections','eligible_image_indices']],
        outputs=[IOSpec(name=n,type_desc='object') for n in ['training_arrays','validation_arrays']]))
    for target,name in [('training_batches','training_arrays'),('validation_batches','validation_arrays')]:
        graph.edges.append(DependencyEdge(source_id='raw_preparation',target_id=target,
            output_name=name,input_name=name,source_type='object',target_type='object'))
    graph.metadata.update(scope='Raw-volume preparation, proposal annotation, detector transfer, consecutive training/validation epochs and inference.',
        remaining_training_obligations=['independent complete graph/source comparison and publication evidence'],
        runtime_requirements=['explicit detector checkpoint','classifier proposal arrays in the preprocessed coordinate frame',
                              'raw arrays, annotations, spatial metadata and explicit sample orders'],
        num_nodes=len(graph.nodes),num_edges=len(graph.edges))
    return graph

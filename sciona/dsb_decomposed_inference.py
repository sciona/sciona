"""Explicit inference graph with reusable tile split/reconstruction bindings."""
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.dsb_decomposed_execution import build_dsb_sampling_graph


def build_dsb_inference_steps_graph():
    graph=build_dsb_sampling_graph()
    graph.nodes=[n for n in graph.nodes if n.node_id!='prediction']
    graph.edges=[e for e in graph.edges if e.target_id!='prediction']
    prefix='sciona.atoms.dl.detection.dsb_inference_steps.'
    spatial='sciona.atoms.numerical.spatial.volume_tiles.'
    def node(identity,primitive,inputs,outputs):
        graph.nodes.append(AlgorithmicNode(node_id=identity,name=identity.replace('_',' '),description=primitive,
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=primitive,
            inputs=[IOSpec(name=n,type_desc=t,required=r) for n,t,r in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,target,output,input_name,kind='object'):
        graph.edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=input_name,
            source_type=next(p.type_desc for n in graph.nodes if n.node_id==source for p in n.outputs if p.name==output),
            target_type=next(p.type_desc for n in graph.nodes if n.node_id==target for p in n.inputs if p.name==input_name)))
    node('inference_geometry',prefix+'prepare_inference_geometry',
        [('processed_volume','numpy.ndarray',True),('tile_side','int',False),('margin','int',False)],
        [('padded','numpy.ndarray'),('coordinates','numpy.ndarray')]+[(n,'int') for n in
         ['side_len','max_stride','margin','coordinate_side','coordinate_stride','coordinate_margin','output_stride']])
    edge('preprocess','inference_geometry','processed_volume','processed_volume','numpy.ndarray')
    for identity,volume,side,stride,margin in [
        ('image_tiles','padded','side_len','max_stride','margin'),
        ('coordinate_tiles','coordinates','coordinate_side','coordinate_stride','coordinate_margin')]:
        node(identity,spatial+'split_volume_tiles',
            [('volume','numpy.ndarray',True),('side_len','int',True),('max_stride','int',True),('margin','int',True)],
            [('tiles','numpy.ndarray'),('grid','object')])
        edge('inference_geometry',identity,volume,'volume','numpy.ndarray')
        for output,input_name in [(side,'side_len'),(stride,'max_stride'),(margin,'margin')]:
            edge('inference_geometry',identity,output,input_name,'int')
    node('normalize_tiles',prefix+'normalize_and_align_tiles',
        [(n,'object',True) for n in ['tiles','image_grid','coordinate_tiles','coordinate_grid']],
        [('tiles','object'),('coordinates','object'),('grid','object')])
    for source,output,target in [('image_tiles','tiles','tiles'),('image_tiles','grid','image_grid'),
                                 ('coordinate_tiles','tiles','coordinate_tiles'),('coordinate_tiles','grid','coordinate_grid')]:
        edge(source,'normalize_tiles',output,target)
    node('detector_tiles',prefix+'predict_detector_tiles',
        [(n,'object',True) for n in ['tiles','coordinates','detector']]+[('tile_batch_size','int',False)], [('predictions','numpy.ndarray')])
    for name in ['tiles','coordinates']:edge('normalize_tiles','detector_tiles',name,name)
    edge('models','detector_tiles','detector','detector')
    node('reconstruct',spatial+'combine_volume_tiles',
        [('tiles','numpy.ndarray',True),('grid','object',True)]+[(n,'int',True) for n in ['side_len','stride','margin']], [('output','object')])
    edge('detector_tiles','reconstruct','predictions','tiles','numpy.ndarray')
    edge('normalize_tiles','reconstruct','grid','grid')
    for output,target in [('side_len','side_len'),('output_stride','stride'),('margin','margin')]:edge('inference_geometry','reconstruct',output,target,'int')
    for identity,function,input_name in [('decode','decode_detector_proposals','output'),
                                         ('filter_proposals','filter_classifier_proposals','proposals'),
                                         ('suppress_proposals','suppress_classifier_proposals','proposals')]:
        node(identity,prefix+function,[(input_name,'object',True)],[('proposals','object')])
    edge('reconstruct','decode','output','output');edge('decode','filter_proposals','proposals','proposals')
    edge('filter_proposals','suppress_proposals','proposals','proposals')
    node('classifier_crops',prefix+'prepare_classifier_crops',
        [('processed_volume','numpy.ndarray',True),('proposals','object',True),('topk','int',False),('crop_size','int',False)],
        [('crops','object'),('coordinates','object')])
    edge('preprocess','classifier_crops','processed_volume','processed_volume','numpy.ndarray')
    edge('suppress_proposals','classifier_crops','proposals','proposals')
    node('prediction',prefix+'predict_classifier_crops',[(n,'object',True) for n in ['crops','coordinates','classifier']],
        [('case_probability','numpy.ndarray'),('proposal_probabilities','numpy.ndarray')])
    for name in ['crops','coordinates']:edge('classifier_crops','prediction',name,name)
    edge('models','prediction','classifier','classifier')
    graph.metadata.update(scope='Explicit inference stages and reusable tiling/sampling; preparation and training decomposition remain.',
        num_nodes=len(graph.nodes),num_edges=len(graph.edges),publication_status='draft')
    return graph

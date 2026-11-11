"""Explicit two-branch inference and source keyed blending, without training."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.osic_image_population_graph import build_image_population_graph
from sciona.osic_keyed_quantile_graph import build_keyed_quantile_graph


def build_combined_inference_graph():
    nodes=[]
    edges=[]
    for prefix, graph in (('image_',build_image_population_graph()),
                          ('quantile_',build_keyed_quantile_graph())):
        for node in graph.nodes:
            nodes.append(node.model_copy(deep=True,update={'node_id':prefix+node.node_id}))
        for edge in graph.edges:
            edges.append(edge.model_copy(deep=True,update={
                'source_id':prefix+edge.source_id,'target_id':prefix+edge.target_id}))
    nodes.append(AlgorithmicNode(node_id='coordinates',name='Route independent response declarations',
        description='Preserve caller declarations for the slope model and response model separately.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.osic_coordinate_routing.route_response_coordinates',
        inputs=[IOSpec(name=name,type_desc='str',constraints=
            'Nonblank exact declaration matching the corresponding private model state. Caller owns units and dimensional compatibility.')
            for name in ('image_response_coordinate','quantile_response_coordinate')],
        outputs=[IOSpec(name=name,type_desc='str',constraints='Exact independent caller declaration; no unit inference or conversion.')
                 for name in ('image','quantile')]))
    nodes.append(AlgorithmicNode(node_id='blend',name='Align keys and blend branch predictions',
        description='Apply the approved source 0.45/0.55 keyed blend to compatible point and uncertainty columns.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.osic_image.blend_keyed_predictions',
        inputs=[IOSpec(name=name,type_desc=kind,constraints=
            'Identical unique key sets and aligned finite (M,2) predictions. Caller establishes common response units and uncertainty meaning.')
            for name,kind in (('image_keys','list[str]'),('image_values','numpy.ndarray'),
                              ('quantile_keys','list[str]'),('quantile_values','numpy.ndarray'))],
        outputs=[IOSpec(name='keys',type_desc='list[str]',constraints='Sorted common keys.'),
                 IOSpec(name='values',type_desc='numpy.ndarray',constraints='Source blended (M,2) values in sorted key order; no calibration claim.')]))
    for source,target,output,input_name,kind in [
        ('coordinates','image_restore','image','response_coordinate','str'),
        ('coordinates','quantile_features','quantile','response_coordinate','str'),
        ('coordinates','quantile_replay','quantile','response_coordinate','str'),
        ('image_population','blend','query_keys','image_keys','list[str]'),
        ('image_reduce','blend','result','image_values','numpy.ndarray'),
        ('quantile_query_route','blend','keys','quantile_keys','list[str]'),
        ('quantile_reduce','blend','result','quantile_values','numpy.ndarray')]:
        edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,
            input_name=input_name,source_type=kind,target_type=kind))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(num_nodes=15,num_edges=26,
        scope='Mapped keyed queries and encoded images through both frozen inference branches and source blending.',
        required_source_policy=dict(dtype='float32',batch_size=32),
        boundary_outputs=[dict(node='blend',port='keys'),dict(node='blend',port='values')],
        caller_obligations=['Compatible slope/time/response units and branch uncertainty meanings',
            'Authentic private states and exact declared transductive query scope'],
        exclusions=['Training and whole-CDG approval','HTTP transport and empirical effectiveness']))

"""One-entity raw slice preparation connected to explicit model-state inference."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.osic_image_lifecycle_graphs import build_prepared_image_inference_graph


def build_image_slice_graph():
    graph=build_prepared_image_inference_graph()
    prepare=AlgorithmicNode(node_id='prepare',name='Prepare selected encoded slices',
        description='Apply explicit source slice selection, decoding and image/metadata preparation for one entity.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.osic_image.prepare_image_slice_inputs',
        inputs=[IOSpec(name=name,type_desc=kind,constraints=bounds) for name,kind,bounds in [
            ('encoded_planes','list','Caller-ordered unique named encoded planes for one entity; at least two slices survive source selection.'),
            ('numeric','float','Caller-mapped finite numeric metadata coordinate; source feature interpretation remains explicit.'),
            ('category','str','Caller-mapped category using source case-sensitive inference encoding.'),
            ('status','str','Caller-mapped status using source case-sensitive inference encoding.')]],
        outputs=[IOSpec(name='result',type_desc='dict',constraints='Prepared image and metadata arrays plus selected positions. No model execution.')])
    select=AlgorithmicNode(node_id='select_inputs',name='Select ordered model inputs',
        description='Borrow prepared values in the explicit caller-selected order.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.pipeline.ordered_fields.select_values',
        inputs=[IOSpec(name='mapping',type_desc='dict',constraints='Prepared values with exact nonblank string field names.'),
                IOSpec(name='fields',type_desc='list[str]',constraints="Explicit ordered names ['images','metadata'] for this source model. Field selection itself does not infer order or validate array semantics.")],
        outputs=[IOSpec(name='result',type_desc='list',constraints='Borrowed prepared image and metadata arrays in caller order; downstream predictor checks numeric validity and model geometry.')])
    graph.nodes.extend([prepare,select])
    graph.edges.extend([
        DependencyEdge(source_id='prepare',target_id='select_inputs',output_name='result',input_name='mapping',source_type='dict',target_type='dict'),
        DependencyEdge(source_id='select_inputs',target_id='predict',output_name='result',input_name='inputs',source_type='list',target_type='list[numpy.ndarray]')])
    graph.metadata.update(scope='One-entity encoded-slice preparation, explicit ordered input selection and restored-model prediction.',
        num_nodes=4,num_edges=3,
        exclusions=['Population iteration and identity routing','Trajectory reduction and branch blending',
                    'Image training and original competition CDG approval','HTTP transport and empirical effectiveness'])
    return graph

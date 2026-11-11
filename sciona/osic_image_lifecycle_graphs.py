"""Explicit prepared-input inference slice of the source image lifecycle."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_prepared_image_inference_graph():
    restore = AlgorithmicNode(
        node_id='restore', name='Restore numeric image model state',
        description='Reconstruct the fixed local model from private numeric inference state without fitting.',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.osic_image_lifecycle.restore_image_prediction_state',
        inputs=[
            IOSpec(name='directory', type_desc='str | pathlib.Path',
                   constraints='Stable caller-owned private numeric-state directory; caller establishes provenance and access.'),
            *[IOSpec(name=name, type_desc='str',
                     constraints='Explicit nonempty coordinate declaration matching saved state exactly; caller establishes meaning and units.')
              for name in ('image_coordinate','metadata_coordinate','response_coordinate')]],
        outputs=[IOSpec(name='result', type_desc='tf_keras.Model',
                        constraints='Uncompiled fixed B5 image/metadata model with validated numeric weights and no optimizer state.')])
    predict = AlgorithmicNode(
        node_id='predict', name='Predict prepared image slices',
        description='Run the restored model on caller-prepared ordered image and metadata arrays.',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.keras_bounded_prediction.predict_bounded',
        inputs=[
            IOSpec(name='model', type_desc='tf_keras.Model', constraints='Restored fixed image model; no fitting.'),
            IOSpec(name='inputs', type_desc='list[numpy.ndarray]',
                   constraints='Ordered [images, metadata], finite float32/float64 arrays shaped (N,512,512,1) and (N,4), N>0. Caller preparation and coordinates must agree with saved state.'),
            IOSpec(name='batch_size', type_desc='int',
                   constraints='Explicit positive Python integer excluding bool. Source inference uses 32; other batch policies require separate source-parity qualification. Reserve one private dataset worker and bound model compute threads.')],
        outputs=[IOSpec(name='result', type_desc='numpy.ndarray',
                        constraints='Finite floating (N,1) scalar slice slopes in caller-declared response units, preserving input order.')])
    edge=DependencyEdge(source_id='restore',target_id='predict',output_name='result',input_name='model',
                        source_type='tf_keras.Model',target_type='tf_keras.Model')
    return CDGExport(nodes=[restore,predict],edges=[edge],metadata=dict(
        scope='Prepared image inference from private numeric state with an explicit restored-model edge.',
        num_nodes=2,num_edges=1,
        exclusions=['Raw image preparation and population iteration','Trajectory reduction and branch blending',
                    'Image training and original competition CDG approval','HTTP transport and empirical effectiveness']))

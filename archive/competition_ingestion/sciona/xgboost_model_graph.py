"""Composable native-model fit/predict graphs with separate population inputs."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_xgboost_model_graph(task):
    if task not in ('regression', 'binary'):
        raise ValueError('Regression or binary graph required')
    target_type = 'NDArray[np.float64]' if task == 'regression' else 'NDArray[np.int64]'
    target_contract = ('Finite float64 vector aligned to training rows; caller-declared target units.' if task == 'regression'
        else 'Int64 vector aligned to training rows; both classes 0 and 1; dimensionless.')
    ports = {
        'features': IOSpec(name='features', type_desc='NDArray[np.float64]', constraints='Nonempty training rows/features matrix; NaN means backend missing value; infinities rejected.'),
        'prediction_features': IOSpec(name='prediction_features', type_desc='NDArray[np.float64]', constraints='Independent nonempty prediction rows/features matrix; same ordered feature columns; NaN accepted, infinities rejected.'),
        'targets': IOSpec(name='targets', type_desc=target_type, constraints=target_contract),
        'feature_names': IOSpec(name='feature_names', type_desc='list', constraints='Unique ordered nonempty feature names matching both matrices; unchanged through model fitting and prediction; feature meanings and units supplied by caller.'),
        'state': IOSpec(name='state', type_desc='dict', constraints='Native UBJ model state with exact task, backend version, feature names and payload digest. Private runtime material for non-public training inputs.'),
        'predictions': IOSpec(name='predictions', type_desc='NDArray[np.float32]', constraints='One finite float32 value per prediction row; regression target units or binary class-1 probability; no rounding, thresholding or calibration.'),
    }
    nodes = [AlgorithmicNode(node_id=node_id, name=function.replace('_', ' '),
        description='Reusable one-thread CPU native-model operation', concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.xgboost.model_io.'+function,
        inputs=[ports[name] for name in inputs], outputs=[ports[output]])
        for node_id, function, inputs, output in [
            ('fit', 'fit_'+task+'_model', ['features', 'targets', 'feature_names'], 'state'),
            ('predict', 'predict_'+task+'_model', ['prediction_features', 'feature_names', 'state'], 'predictions')]]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='fit', target_id='predict',
        output_name='state', input_name='state', source_type='dict', target_type='dict')],
        metadata=dict(artifact_source='reusable_computation_reconstruction', publication_status='draft',
            scope='Native '+task+' model fitting and prediction on separately supplied populations.',
            source_version_ids=['7d6f7b78-1dff-5757-9326-4edad55aaa68'],
            source_content_hashes=['5a9856b1f6a4db73e7ce12809248de193b63ad06401a6e73b8322744cb7c44bd'],
            exclusions=['Complete NASA workflow', 'Automatic feature engineering or selection', 'Empirical domain validation', 'Historical backend identity'],
            num_nodes=2, num_edges=1))

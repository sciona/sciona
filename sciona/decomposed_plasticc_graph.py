"""Complete explicit PLAsTiCC lifecycle with separately bound numeric atoms.

Draft graph only. Registered function signatures define ports; qualification
must pin both the serialized graph and exact provider implementations.
"""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.ml import decomposed_plasticc_numeric as numeric
from sciona.atoms.ml import decomposed_plasticc_lifecycle as lifecycle
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix


NUMERIC_OUTPUTS = dict(initial_amplitude=['amplitude'], fit_gp=['fitted'], scale_covariance=['covariance'],
    observation_variance=['variance'], condition=['factor', 'alpha', 'logdet'], predict_gp=['mean', 'variance'],
    balanced_weights=['weights'], reference_stratified_folds=['folds'], normalize_scores=['probabilities'],
    predict_scores=['scores'], average_folds=['probabilities'])
LIFECYCLE_OUTPUTS = dict(prepare=['prepared'], augment=['population'], featurizer=['featurizer'],
    extract_features=['features'], assign_folds=['folds'], training_weights=['weights', 'class_weights'],
    training_inputs=['labels', 'parameters'], fit_fold=['model'], fit_folds=['result'], complete=['completed'],
    prediction_inputs=['population', 'featurizer', 'models', 'classes'], task_policy=['final'], encode=['result'])


def build_decomposed_plasticc_graph():
    nodes = []; edges = []
    def add(name, fn, outputs, bindings):
        signature = inspect.signature(fn)
        def typename(annotation):
            return getattr(annotation, '__name__', str(annotation))
        inputs = [IOSpec(name=p.name, type_desc=typename(p.annotation), required=p.default is inspect.Parameter.empty,
                    default_value_repr='' if p.default is inspect.Parameter.empty else repr(p.default))
                  for p in signature.parameters.values()]
        # Match the already-published canonical kernel interface, rather than
        # exposing its postponed Python annotation aliases as catalog types.
        if fn is matern_kernel_matrix:
            for port, kind in zip(inputs, ('ndarray', 'object', 'object', 'float'), strict=True):
                port.type_desc = kind
        returned = signature.return_annotation
        types = returned.__args__ if getattr(returned, '__origin__', None) is tuple else (returned,)
        if fn is matern_kernel_matrix:
            types = ('ndarray',)
        if len(types) != len(outputs):
            raise ValueError('Output annotation count differs')
        nodes.append(AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(fn) or name,
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=fn.__module__+'.'+fn.__name__,
            inputs=inputs, outputs=[IOSpec(name=n, type_desc=typename(t)) for n, t in zip(outputs, types)]))
        for port in inputs:
            if port.name not in bindings:
                continue
            source, output, callable_input = bindings[port.name]
            edges.append(DependencyEdge(source_id=source, target_id=name, output_name=output, input_name=port.name,
                source_type='object' if callable_input else port.type_desc, target_type=port.type_desc,
                edge_kind=EdgeKind.CALLABLE_INJECTION if callable_input else EdgeKind.DATA_FLOW))
    def data(node, port): return (node, port, False)
    def function(node, port): return (node, port, True)
    for operation in NUMERIC_OUTPUTS:
        if operation != 'average_folds':
            add(operation, getattr(numeric, operation), NUMERIC_OUTPUTS[operation], {})
    add('kernel', matern_kernel_matrix, ['result'], {})
    add('fit_fold', lifecycle.fit_fold, ['model'], {})
    def stage(name, operation, bindings):
        add(name, getattr(lifecycle, operation), LIFECYCLE_OUTPUTS[operation], bindings)
    stage('prepare', 'prepare', dict(
        amplitude_function=function('initial_amplitude', 'amplitude'), fit_function=function('fit_gp', 'fitted'),
        kernel_function=function('kernel', 'result'), scale_function=function('scale_covariance', 'covariance'),
        noise_function=function('observation_variance', 'variance'), condition_function=function('condition', 'factor'),
        predict_function=function('predict_gp', 'mean')))
    prepared = data('prepare', 'prepared'); population = data('augment', 'population')
    stage('augment', 'augment', dict(prepared=prepared))
    stage('featurizer', 'featurizer', dict(prepared=prepared))
    policy = data('featurizer', 'featurizer')
    stage('training_features', 'extract_features', dict(population=population, featurizer=policy))
    stage('folds', 'assign_folds', dict(population=population, prepared=prepared,
        fold_function=function('reference_stratified_folds', 'folds')))
    stage('weights', 'training_weights', dict(population=population, prepared=prepared,
        balance_function=function('balanced_weights', 'weights')))
    stage('training_inputs', 'training_inputs', dict(population=population, prepared=prepared))
    stage('fit_folds', 'fit_folds', dict(features=data('training_features', 'features'),
        labels=data('training_inputs', 'labels'), weights=data('weights', 'weights'), folds=data('folds', 'folds'),
        parameters=data('training_inputs', 'parameters'), fit_function=function('fit_fold', 'model')))
    stage('complete', 'complete', dict(prepared=prepared, population=population, featurizer=policy,
        result=data('fit_folds', 'result'), weights=data('weights', 'weights'), class_weights=data('weights', 'class_weights')))
    stage('prediction_inputs', 'prediction_inputs', dict(completed=data('complete', 'completed')))
    query = data('prediction_inputs', 'population')
    stage('query_features', 'extract_features', dict(population=query, featurizer=data('prediction_inputs', 'featurizer')))
    add('average_folds', numeric.average_folds, ['probabilities'], dict(models=data('prediction_inputs', 'models'),
        features=data('query_features', 'features'), classes=data('prediction_inputs', 'classes'),
        score_operation=function('predict_scores', 'scores'), normalize_operation=function('normalize_scores', 'probabilities')))
    stage('task_policy', 'task_policy', dict(population=query, probabilities=data('average_folds', 'probabilities')))
    stage('encode', 'encode', dict(final=data('task_policy', 'final')))
    return CDGExport(nodes=nodes, edges=edges, metadata=dict(publication_status='draft',
        artifact_source='complete_original_competition_reconstruction',
        source_version_id='57909e28-f500-51c8-91da-8e104377d2fd',
        catalog_output_sources={'result': dict(node_id='encode', output_name='result')},
        num_nodes=len(nodes), num_edges=len(edges),
        scope='Every configured augmentation, feature extraction, stratified fold, model and prediction executes.',
        reuse_boundary='Generic GP, class balancing, reference-preserving folds and ensemble numerics are independently bound. Astronomy features, redshift weighting and final class policy are explicit.',
        limits=['Draft graph; provider and original catalog qualification remain pending.',
                'Synthetic execution does not establish historical competition accuracy.']))

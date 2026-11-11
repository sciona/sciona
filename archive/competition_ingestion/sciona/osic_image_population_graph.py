"""Sequential population graph with explicit callable and numerical boundaries."""
from sciona.architect.models import AlgorithmicNode,DependencyEdge,EdgeKind,IOSpec,NodeStatus
from sciona.osic_image_slice_graph import build_image_slice_graph


def build_image_population_graph():
    graph=build_image_slice_graph()
    graph.edges=[]
    ports=[('baselines','pandas.DataFrame'),('queries','pandas.DataFrame'),('planes','dict'),
           ('model','tf_keras.Model'),('batch_size','int'),('prepare_function','object'),
           ('select_function','object'),('predict_function','object')]
    outputs=[('slice_slopes','list'),('baseline_time','numpy.ndarray'),('baseline_value','numpy.ndarray'),
             ('baseline_uncertainty','numpy.ndarray'),('query_subject','numpy.ndarray'),
             ('query_time','numpy.ndarray'),('query_keys','list[str]'),('selected_positions','list')]
    dispatch=AlgorithmicNode(node_id='population',name='Route entities sequentially',
        description='Validate mapped populations, invoke explicit computations per entity and collect only scalar slopes and routing coordinates.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.osic_image_routing.route_and_predict',
        inputs=[IOSpec(name=name,type_desc=typ,constraints=
            'Explicit source-mapped population and trusted execution bindings. Batch size must be 32; preserve baseline and query order and full entity coverage.') for name,typ in ports],
        outputs=[IOSpec(name=name,type_desc=typ,constraints=
            'Validated population routing output; preserve order, scalar prediction dtype and caller coordinate meanings.') for name,typ in outputs])
    reduce=AlgorithmicNode(node_id='reduce',name='Reduce slice slopes to trajectories',
        description='Apply the separate source median, intercept and uncertainty policy to aligned query coordinates.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.osic_image.reduce_image_predictions',
        inputs=[IOSpec(name=name,type_desc=typ,constraints=
            'Exact routed scalar predictions and aligned coordinates. Caller ensures compatible time and slope units.') for name,typ in outputs[:6]],
        outputs=[IOSpec(name='result',type_desc='numpy.ndarray',constraints=
            'Point and source heuristic uncertainty (M,2) in original query-key order. Negative uncertainty is retained; no calibration claim.')])
    graph.nodes.extend([dispatch,reduce])
    graph.edges.append(DependencyEdge(source_id='restore',target_id='population',output_name='result',
        input_name='model',source_type='tf_keras.Model',target_type='tf_keras.Model'))
    for source,target in [('prepare','prepare_function'),('select_inputs','select_function'),('predict','predict_function')]:
        graph.edges.append(DependencyEdge(source_id=source,target_id='population',output_name='result',
            input_name=target,source_type='object',target_type='object',edge_kind=EdgeKind.CALLABLE_INJECTION))
    for name,typ in outputs[:6]:
        graph.edges.append(DependencyEdge(source_id='population',target_id='reduce',output_name=name,
            input_name=name,source_type=typ,target_type=typ))
    graph.metadata.update(scope='Raw mapped image population inference with explicit numeric state, callable operations and separate trajectory reduction.',
        num_nodes=6,num_edges=10,
        boundary_outputs=[dict(node='population',port='query_keys'),dict(node='reduce',port='result'),
                          dict(node='population',port='selected_positions')],
        exclusions=['Quantile branch and final blending','Complete image training and original CDG approval',
                    'HTTP transport and empirical effectiveness'])
    return graph

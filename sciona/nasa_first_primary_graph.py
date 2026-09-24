"""Decomposed raw-input primary prediction with explicit source arithmetic."""
import sciona.atoms.ml.tabular.available_features as features
from sciona.available_feature_contracts import contracts
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.nasa_first_feature_graph import build_nasa_first_feature_graph


def build_nasa_first_primary_graph():
    graph = build_nasa_first_feature_graph()
    generic = 'sciona.atoms.ml.tabular.available_features.'
    prediction = 'sciona.atoms.ml.model_selection.materialized_prediction.'
    domain = 'sciona.atoms.ml.domain_adapters.first_place_prediction.'
    def port(name, typ, constraint):
        return IOSpec(name=name, type_desc=typ, constraints=constraint)
    def add(identity, runtime, inputs, outputs):
        graph.nodes.append(AlgorithmicNode(node_id=identity, name=runtime.rsplit('.',1)[-1],
            description='Decomposed primary prediction operation', concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=runtime, inputs=inputs, outputs=outputs))
    def connect(source, output, target, name):
        before = next(p for n in graph.nodes if n.node_id == source for p in n.outputs if p.name == output)
        after = next(p for n in graph.nodes if n.node_id == target for p in n.inputs if p.name == name)
        graph.edges.append(DependencyEdge(source_id=source, target_id=target, output_name=output, input_name=name,
            source_type=before.type_desc, target_type=after.type_desc))
    def generic_node(identity, symbol):
        inputs, outputs = contracts(symbol, getattr(features, symbol))
        add(identity, generic+symbol, [IOSpec(**p) for p in inputs], [IOSpec(**p) for p in outputs])
    frame = lambda name: port(name, 'pd.DataFrame', 'Ordered materialized named model features; preserve query row order and declared units.')
    model = lambda name: port(name, 'object', 'Private runtime predictor with ordered feature_names_ and predict(frame, thread_count=1).')
    vector = lambda name: port(name, 'NDArray[np.float64]', 'Finite row-aligned prediction vector in minutes; no broadcasting.')
    integer_vector = lambda name: port(name, 'NDArray[np.int64]', 'Clipped, truncated row-aligned integer predictions in minutes; no broadcasting.')
    population = port('population', 'str', 'Explicit public-software population label for global feature and override routing.')
    generic_node('model_features', 'prepare_features')
    connect('assemble', 'result', 'model_features', 'frame')
    selected = []
    for slot in ('0', '1', '2', 'global'):
        selected.extend([model('model_'+slot), frame('frame_'+slot)])
    selected.extend([vector('baseline'), vector('zeros')])
    selected.extend(port(name, 'int', 'Exact source clipping bound in minutes.') for name in ['local_lower','local_upper','final_lower','final_upper'])
    add('prediction_inputs', domain+'prediction_inputs', [frame('frame'),
        port('models','dict','Explicit slots 0/1/2 and global_model; residual aliases must be preserved.'), population], selected)
    connect('model_features', 'result', 'prediction_inputs', 'frame')
    for slot in ('0', '1', '2', 'global'):
        identity = 'predict_'+slot
        add(identity, prediction+'predict', [model('model'), frame('frame')], [vector('values')])
        connect('prediction_inputs', 'model_'+slot, identity, 'model')
        connect('prediction_inputs', 'frame_'+slot, identity, 'frame')
        if slot != 'global':
            clipped = 'clip_'+slot
            generic_node(clipped, 'clip_truncate')
            connect(identity, 'values', clipped, 'values')
            connect('prediction_inputs', 'baseline' if slot in ('0','2') else 'zeros', clipped, 'offset')
            connect('prediction_inputs', 'local_lower', clipped, 'lower')
            connect('prediction_inputs', 'local_upper', clipped, 'upper')
    vectors = port('vectors', 'list', 'Ordered finite aligned prediction vectors; equal weights and common target units.')
    add('ensemble_inputs', domain+'ensemble_inputs', [integer_vector('local0'),integer_vector('local1'),integer_vector('local2'),vector('global_values'),population], [vectors])
    for slot in ('0','1','2'):
        connect('clip_'+slot, 'result', 'ensemble_inputs', 'local'+slot)
    connect('predict_global', 'values', 'ensemble_inputs', 'global_values')
    add('average', prediction+'average', [vectors], [vector('values')])
    connect('ensemble_inputs', 'vectors', 'average', 'vectors')
    generic_node('final_clip', 'clip_truncate')
    connect('average','values','final_clip','values')
    for source, target in [('zeros','offset'),('final_lower','lower'),('final_upper','upper')]:
        connect('prediction_inputs', source, 'final_clip', target)
    add('format', domain+'format_predictions', [
        port('queries','pd.DataFrame','Caller query table with original identities, row order and duplicate index retained.'),
        integer_vector('values')], [port('predictions','pd.DataFrame','Owned query table with aligned int64 primary predictions in minutes.')])
    connect('final_clip', 'result', 'format', 'values')
    graph.metadata = dict(artifact_source='decomposed_first_place_primary', publication_status='draft',
        scope='Raw inputs through all corrected feature families, fixed model policy, four native predictions and exact source clipping/ensemble arithmetic.',
        exclusions=['Model training', 'Baseline/constant fallback routing', 'Empty-query short circuit', 'Static symbolic propagation'])
    return graph

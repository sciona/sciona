"""Draft decomposed domain feature graph around the reusable numerical graph."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.residual_classifier_graph import build_residual_classifier_graph


def domain_contract(function, name, output=False):
    """Public software interface contracts; never inferred from runtime datasets."""
    if function == 'numerical_handoff':
        if name in ('features','prediction_features'):
            return ('Materialized nonempty float64 matrix with persisted feature order; training and query rows '
                    'may differ. NaN uses backend missing behavior; infinity rejected. Observed shapes are checked '
                    'through all numerical witnesses before fitting, not inferred before the domain joins.')
        controls = {
            'train_fraction':'Finite fraction strictly between zero and one, applied to unique training groups.',
            'seed':'Unsigned 32-bit integer seed for the grouped split.',
            'maximum_error':'Finite nonnegative residual threshold in minutes, sharing the training target scale.',
            'threshold':'Finite dimensionless probability threshold in [0,1]; both calibration groups must later be populated.',
            'prediction_feature_name':'Explicit new nonempty backend-compatible feature name, absent from the trained base schema.',
        }
        if name in controls: return controls[name]
    queries = ('Nonempty DataFrame with unique (gufi,timestamp) rows, nonempty string identities, '
               'one four-character airport and timezone-naive datetime64 query timestamps in the caller clock. '
               'Preserve row identity and order. Training queries additionally carry numeric minutes_until_pushback.')
    departures = ('DataFrame keyed uniquely by (gufi,timestamp), with remaining-time, estimate-age, mean, median, '
                  'population-standard-deviation and count features from eligible departure estimates. '
                  'Times are minutes. Eligibility requires observation time strictly before query and estimate '
                  'strictly after query; latest observation wins with stable input-order ties. No eligible estimates '
                  'means no feature row; final and singleton entities are included.')
    categories = ('One row per gufi with its airport and fixed indicator columns: persisted three-character '
                  'vocabulary plus an always-present Other fallback. No inference refit; unknown prefixes use Other.')
    history = ('One row per qualifying unique query timestamp with mean and population standard deviation '
               'of completed nonnegative arrival-to-stand durations in minutes. Both streams must be observed '
               'before the query; use latest complete snapshots and reject ambiguous equal-time revisions. '
               'Destination matches the airport suffix. Strict past-window policy starts at 60 minutes, expands '
               'once to 180 for zero observations or 120 for fewer than three. Empty history yields no row.')
    if name in ('training_records', 'prediction_records'):
        return ('Runtime dict with exactly queries, estimates, stands and arrivals DataFrames. No filesystem '
                'loading, implicit source discovery or retained records. '+queries)
    if name == 'airport':
        return 'Explicit four-character airport identifier matching every query row; final three characters select arrival destination.'
    if name == 'queries': return queries
    if name == 'estimates':
        return ('DataFrame with gufi, timestamp and departure_runway_estimated_time; both clocks are timezone-naive '
                'datetime64 in the caller clock, with no missing required values. Unqueried entities are allowed.')
    if name == 'stands':
        return ('DataFrame with gufi, timestamp observation time and arrival_stand_actual_time event time; '
                'timezone-naive datetime64 in the caller clock. Event streams may include non-query entities.')
    if name == 'arrivals':
        return ('DataFrame with gufi, timestamp observation time and arrival_runway_actual_time event time; '
                'timezone-naive datetime64 in the caller clock. Identity encodes the source destination convention.')
    if name == 'vocabulary':
        return ('Immutable tuple of unique three-character strings. Fit once from unique training entities: '
                'up to 25 prefixes by decreasing frequency, lexical tie break. Reuse unchanged at inference.')
    if name == 'departures' or function == 'departure_features' and output: return departures
    if name == 'categories' or function == 'category_features' and output: return categories
    if name == 'history' or function == 'history_features' and output: return history
    if name == 'frame':
        training = function in ('join_training', 'training_arrays')
        return ('Joined feature DataFrame with unix_time in integer seconds and declared ETD, category and history columns. '
                + ('Training: inner departure join, left category/history joins, then drop rows with any missing value. '
                   'Surviving row count is data-dependent and must be nonzero for array extraction; targets are finite numeric minutes.'
                   if training else 'Prediction: left joins preserve every query, with NaN for missing features. '
                   'Array conversion restores query identity order and uses the persisted training column schema.'))
    if name == 'features':
        return ('Float64 row-by-feature matrix with columns in the accompanying persisted feature_names order. '
                + ('Training values are finite; rows align exactly with targets and groups.' if function=='training_arrays'
                   else 'Prediction rows align with output identities; NaNs use backend missing-value behavior, infinities and nonnumeric columns are rejected.'))
    if name == 'targets': return 'Finite float64 vector in minutes, aligned to training feature rows; no target scaling or imputation.'
    if name == 'groups': return 'Int64 group codes from sorted unique gufi values, aligned to training rows; repeated entities retain group membership.'
    if name == 'feature_names':
        return ('Unique ordered nonempty backend-compatible list: unix_time, four ETD features, Other, persisted '
                'vocabulary indicators, and two history features. No unordered-set schema or inference refit.')
    if name == 'identities': return queries+' Exactly gufi,timestamp,airport columns in original query order.'
    if name == 'predictions': return 'One int32 nearest-even quantized minute prediction per identity row; order and length must match exactly.'
    if name == 'prediction_table': return 'Copy of query identity columns with aligned int32 minutes_until_pushback values; no input mutation or row reordering.'
    raise ValueError('Missing domain contract: '+function+'/'+name)


def build_nasa_domain_graph():
    core = build_residual_classifier_graph()
    nodes, edges = [], list(core.edges)
    prefix = 'sciona.atoms.ml.domain_adapters.airport_features.'
    frame = 'pd.DataFrame'
    array = 'NDArray[np.float64]'
    def add(identity, function, inputs, outputs):
        ports = []
        for name, (typ, source, output) in inputs.items():
            ports.append(IOSpec(name=name, type_desc=typ, constraints=domain_contract(function,name)))
            if source is not None:
                edges.append(DependencyEdge(source_id=source, target_id=identity, output_name=output,
                    input_name=name, source_type=typ, target_type=typ))
        nodes.append(AlgorithmicNode(node_id=identity, name=function.replace('_',' '),
            description='Corrected airport domain adapter; numerical computation remains in shared atoms.',
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=prefix+function,
            inputs=ports, outputs=[IOSpec(name=name,type_desc=typ,
                constraints=domain_contract(function,name,output=True)) for name,typ in outputs.items()]))
    def root(typ): return typ, None, None
    def port(typ, node, name): return typ, node, name
    for phase, function, key in [('domain_train','unpack_training','training_records'),
                                  ('domain_query','unpack_prediction','prediction_records')]:
        add(phase, function, {key:root('dict'), 'airport':root('str')},
            dict(queries=frame, estimates=frame, stands=frame, arrivals=frame))
    add('domain_vocabulary','fit_categories',dict(queries=port(frame,'domain_train','queries')),dict(vocabulary='tuple'))
    for phase in ['domain_train','domain_query']:
        query = port(frame,phase,'queries')
        add(phase+'_departures','departure_features',dict(queries=query,estimates=port(frame,phase,'estimates')),dict(features=frame))
        add(phase+'_categories','category_features',dict(queries=query,vocabulary=port('tuple','domain_vocabulary','vocabulary')),dict(features=frame))
        add(phase+'_history','history_features',dict(queries=query,stands=port(frame,phase,'stands'),arrivals=port(frame,phase,'arrivals'),airport=root('str')),dict(features=frame))
        add(phase+'_join','join_training' if phase=='domain_train' else 'join_prediction',dict(queries=query,
            departures=port(frame,phase+'_departures','features'),categories=port(frame,phase+'_categories','features'),
            history=port(frame,phase+'_history','features')),dict(frame=frame))
    add('domain_train_arrays','training_arrays',dict(frame=port(frame,'domain_train_join','frame'),
        vocabulary=port('tuple','domain_vocabulary','vocabulary')),
        dict(features=array,targets=array,groups='NDArray[np.int64]',feature_names='list'))
    add('domain_query_arrays','prediction_arrays',dict(frame=port(frame,'domain_query_join','frame'),
        queries=port(frame,'domain_query','queries'),feature_names=port('list','domain_train_arrays','feature_names')),
        dict(features=array,identities=frame))
    handoff = {name:port(typ,'domain_train_arrays',name) for name,typ in
               [('features',array),('targets',array),('groups','NDArray[np.int64]'),('feature_names','list')]}
    handoff['prediction_features'] = port(array,'domain_query_arrays','features')
    control_types = dict(train_fraction='float',seed='int',maximum_error='float',threshold='float',prediction_feature_name='str')
    handoff.update({name:root(typ) for name,typ in control_types.items()})
    add('domain_handoff','numerical_handoff',handoff,{name:value[0] for name,value in handoff.items()})
    for name, typ in [('features',array),('targets',array),('groups','NDArray[np.int64]'),('feature_names','list')]:
        edges.append(DependencyEdge(source_id='domain_handoff',target_id='training',output_name=name,input_name=name,source_type=typ,target_type=typ))
    edges.append(DependencyEdge(source_id='domain_handoff',target_id='query',output_name='prediction_features',input_name='prediction_features',source_type=array,target_type=array))
    connected = {(edge.target_id,edge.input_name) for edge in core.edges}
    for node in core.nodes:
        for spec in node.inputs:
            if spec.name in control_types and (node.node_id,spec.name) not in connected:
                typ = control_types[spec.name]
                edges.append(DependencyEdge(source_id='domain_handoff',target_id=node.node_id,output_name=spec.name,
                    input_name=spec.name,source_type=typ,target_type=typ))
    nodes.extend(core.nodes)
    add('domain_output','attach_output',dict(identities=port(frame,'domain_query_arrays','identities'),
        predictions=port('NDArray[np.int32]','final','predictions')),dict(prediction_table=frame))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(core.metadata,
        num_nodes=len(nodes),num_edges=len(edges),scope='Decomposed corrected single-airport domain feature preparation and residual-classifier execution.',
        exclusions=['Whole-graph symbolic table propagation','Multi-airport graph dispatch','Persistent training lifecycle','Original uncorrected source behavior','Empirical competition quality'],
        publication_status='draft'))

"""Complete decomposed source lifecycle with explicit update/checkpoint binding."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,EdgeKind,IOSpec,NodeStatus

STAGES = [
    ('unpack', [('payload', 'dict')], [('raw_frame', 'object'), ('config', 'object'), ('valid_threshold', 'float'), ('add_days', 'int'), ('corr_backoffset', 'int'), ('lags', 'list'), ('thresholds', 'list'), ('output_rows', 'list'), ('output_columns', 'list'), ('identifiers', 'ndarray')], {}),
    ('prepare', [('frame', 'object'), ('valid_threshold', 'float'), ('add_days', 'int'), ('corr_backoffset', 'int')], [('logged', 'object'), ('missing', 'object'), ('starts', 'ndarray'), ('ends', 'ndarray')], {'frame': ('unpack', 'raw_frame'), 'valid_threshold': ('unpack', 'valid_threshold'), 'add_days': ('unpack', 'add_days'), 'corr_backoffset': ('unpack', 'corr_backoffset')}),
    ('feature_axes', [('logged', 'object')], [('numeric_values', 'ndarray'), ('row_labels', 'list'), ('begin', 'object'), ('end', 'object')], {'logged': ('prepare', 'logged')}),
    ('correlations', [('values', 'ndarray'), ('starts', 'ndarray'), ('ends', 'ndarray'), ('lags', 'list'), ('thresholds', 'list'), ('backoffset', 'int')], [('correlation_columns', 'ndarray')], {'values': ('feature_axes', 'numeric_values'), 'starts': ('prepare', 'starts'), 'ends': ('prepare', 'ends'), 'lags': ('unpack', 'lags'), 'thresholds': ('unpack', 'thresholds'), 'backoffset': ('unpack', 'corr_backoffset')}),
    ('popularity', [('logged', 'object')], [('popularity_values', 'object')], {'logged': ('prepare', 'logged')}),
    ('page_metadata', [('row_labels', 'list')], [('page_map', 'ndarray'), ('encoded', 'dict')], {'row_labels': ('feature_axes', 'row_labels')}),
    ('calendar', [('begin', 'object'), ('end', 'object'), ('add_days', 'int')], [('dow', 'ndarray'), ('lagged_ix', 'ndarray'), ('features_end', 'object')], {'begin': ('feature_axes', 'begin'), 'end': ('feature_axes', 'end'), 'add_days': ('unpack', 'add_days')}),
    ('assemble', [('logged', 'object'), ('missing', 'object'), ('correlation_columns', 'ndarray'), ('page_map', 'ndarray'), ('encoded', 'dict'), ('popularity_values', 'object'), ('dow', 'ndarray'), ('lagged_ix', 'ndarray'), ('features_end', 'object')], [('tensors', 'dict'), ('plain', 'dict')], {'logged': ('prepare', 'logged'), 'missing': ('prepare', 'missing'), 'correlation_columns': ('correlations', 'correlation_columns'), 'page_map': ('page_metadata', 'page_map'), 'encoded': ('page_metadata', 'encoded'), 'popularity_values': ('popularity', 'popularity_values'), 'dow': ('calendar', 'dow'), 'lagged_ix': ('calendar', 'lagged_ix'), 'features_end': ('calendar', 'features_end')}),
    ('configure', [('tensors', 'dict'), ('plain', 'dict'), ('config', 'object')], [('prepared', 'object')], {'tensors': ('assemble', 'tensors'), 'plain': ('assemble', 'plain'), 'config': ('unpack', 'config')}),
    ('initialize', [('prepared', 'object')], [('initialized', 'object')], {'prepared': ('configure', 'prepared')}),
    ('advance', [('state', 'dict'), ('batches', 'list'), ('observations', 'list')], [('state', 'dict'), ('metrics', 'list')], {}),
    ('save_checkpoint', [('checkpoints', 'list'), ('state', 'dict')], [('checkpoints', 'list')], {}),
    ('train', [('prepared', 'object'), ('initialized', 'object'), ('advance_function', 'object'), ('checkpoint_function', 'object')], [('trained', 'dict')], {'prepared': ('configure', 'prepared'), 'initialized': ('initialize', 'initialized'), 'advance_function': ('advance', 'state'), 'checkpoint_function': ('save_checkpoint', 'checkpoints')}),
    ('prediction_batches', [('prepared', 'object')], [('records', 'list')], {'prepared': ('configure', 'prepared')}),
    ('checkpoint_forecast', [('trained', 'dict'), ('records', 'list'), ('row_labels', 'list')], [('frame', 'object')], {'trained': ('train', 'trained'), 'records': ('prediction_batches', 'records'), 'row_labels': ('feature_axes', 'row_labels')}),
    ('date_forecast', [('prepared', 'object'), ('frame', 'object')], [('frame', 'object')], {'prepared': ('configure', 'prepared'), 'frame': ('checkpoint_forecast', 'frame')}),
    ('forecast_axes', [('frame', 'object')], [('values', 'ndarray'), ('row_labels', 'list'), ('column_labels', 'list')], {'frame': ('date_forecast', 'frame')}),
    ('aligned_output', [('values', 'ndarray'), ('row_labels', 'list'), ('column_labels', 'list'), ('output_rows', 'list'), ('output_columns', 'list'), ('identifiers', 'ndarray')], [('identifiers', 'ndarray'), ('values', 'ndarray')], {'values': ('forecast_axes', 'values'), 'row_labels': ('forecast_axes', 'row_labels'), 'column_labels': ('forecast_axes', 'column_labels'), 'output_rows': ('unpack', 'output_rows'), 'output_columns': ('unpack', 'output_columns'), 'identifiers': ('unpack', 'identifiers')}),
    ('result', [('identifiers', 'ndarray'), ('values', 'ndarray')], [('result', 'dict')], {'identifiers': ('aligned_output', 'identifiers'), 'values': ('aligned_output', 'values')}),
]


def build_decomposed_webtraffic_graph():
    nodes=[];edges=[]
    callable_inputs={'advance_function','checkpoint_function'}
    for name,inputs,outputs,bindings in STAGES:
        nodes.append(AlgorithmicNode(node_id=name,name=name,description='Explicit '+name.replace('_',' '),
            concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.dl.decomposed_webtraffic_execution.'+name,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
        for port,kind in inputs:
            if port not in bindings:continue
            source,output=bindings[port]
            edges.append(DependencyEdge(source_id=source,target_id=name,output_name=output,input_name=port,
                source_type='object' if port in callable_inputs else kind,target_type=kind,
                edge_kind=EdgeKind.CALLABLE_INJECTION if port in callable_inputs else EdgeKind.DATA_FLOW))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        artifact_source='complete_original_competition_reconstruction',publication_status='draft',
        source_version_id='a6298d16-4fa7-5577-b559-7c17c006ccd5',
        catalog_output_sources={'result':dict(node_id='result',output_name='result')},
        num_nodes=len(nodes),num_edges=len(edges),
        scope='Complete source feature preparation, three-model GRU training, Adam/EMA checkpoints, count-space ensemble and caller-aligned final records.',
        reuse_boundary='Numeric support correlations, popularity normalization and integer output alignment have independent inputs. Page/calendar parsing, three-model lifecycle and source feature layout retain explicit recipe assumptions.',
        limits=['All configured updates and retained checkpoints are required; no forward-only substitute.',
                'CPU RNG streams and explicit EMA observation phases remain declared adaptations.',
                'Single-use mutable streams require fresh initialization for each invocation.',
                'No original TensorFlow checkpoint codec, historical training duration or accuracy claim.',
                'Provider and original publication remain pending.']))

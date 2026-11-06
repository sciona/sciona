"""Complete forecasting computation with visible reusable rolling dependency."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,EdgeKind,IOSpec,NodeStatus


def build_decomposed_future_sales_graph():
    def node(name,inputs,outputs,fqdn=None):
        return AlgorithmicNode(node_id=name,name=name,description='Forecasting '+name,
            concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_future_sales_execution.'+name,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs])
    controls=[('config','dict'),('categorical_columns','list'),('lower','float'),('upper','float')]
    nodes=[
        node('unpack',[('payload','dict')],[('transactions','list'),('entities','list'),('boundaries','list'),('lags','list'),('rolling_windows','list'),('calendar_cycle','int'),('cold_start','float'),('config','dict'),('lower','float'),('upper','float'),('categorical_columns','list')]),
        node('aggregate',[('transactions','list'),('entities','list'),('boundaries','list')],[('totals','ndarray'),('groups','ndarray')]),
        node('bounded_targets',[('totals','ndarray'),('lower','float'),('upper','float')],[('targets','ndarray')]),
        node('rolling',[('values','ndarray'),('window_size','int'),('agg_fns','list')],[('statistics','ndarray')],
             'sciona.atoms.ml.tabular.gradient_boosting.atoms.rolling_statistics'),
        node('lag_features',[('targets','ndarray'),('lags','list'),('rolling_windows','list'),('rolling_function','object')],[('lagged','ndarray')]),
        node('encoding_features',[('targets','ndarray'),('groups','ndarray'),('calendar_cycle','int'),('cold_start','float')],[('encoded','ndarray')]),
        node('assemble',[('targets','ndarray'),('lagged','ndarray'),('encoded','ndarray'),('lags','list'),('rolling_windows','list')],[('panel','object')]),
        node('search',[('panel','object'),*controls],[('selection','dict')]),
        node('refit',[('panel','object'),*controls,('selection','dict')],[('result','dict')])]
    edges=[]
    def link(source,target,name,kind,output=None):
        edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output or name,input_name=name,source_type=kind,target_type=kind))
    for n in ('transactions','entities','boundaries'):link('unpack','aggregate',n,'list')
    link('aggregate','bounded_targets','totals','ndarray')
    for n in ('lower','upper'):link('unpack','bounded_targets',n,'float')
    for target in ('lag_features','encoding_features','assemble'):link('bounded_targets',target,'targets','ndarray')
    for target in ('lag_features','assemble'):
        for n in ('lags','rolling_windows'):link('unpack',target,n,'list')
    link('aggregate','encoding_features','groups','ndarray')
    link('unpack','encoding_features','calendar_cycle','int');link('unpack','encoding_features','cold_start','float')
    link('lag_features','assemble','lagged','ndarray');link('encoding_features','assemble','encoded','ndarray')
    for target in ('search','refit'):
        link('assemble',target,'panel','object')
        for n,t in controls:link('unpack',target,n,t)
    link('search','refit','selection','dict')
    edges.append(DependencyEdge(source_id='rolling',target_id='lag_features',output_name='statistics',input_name='rolling_function',source_type='object',target_type='object',edge_kind=EdgeKind.CALLABLE_INJECTION))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        artifact_source='competition_generic_execution_reconstruction',publication_status='draft',
        source_version_ids=['e6803752-90ef-5cab-8b30-5ebeec716c15'],num_nodes=len(nodes),num_edges=len(edges),
        catalog_output_sources={'result':dict(node_id='refit',output_name='result')},
        scope='Complete configured net aggregation, clipping, causal lag and encoding branch, TPE search, early stopping, final refit and one-period forecast.',
        reuse_boundary='Numeric grouping widths, training feature widths, categorical columns and prediction bounds are independent provider inputs. Root recipe selects three categorical slots and [0,20] bounds. Rolling is an explicit generic callable dependency.',
        limits=['Separate fresh single-thread workers for search and refit; native combined-process crash is not claimed resolved.',
                'Source explicitly permits alternative encodings. No holiday, promotion, embedding or historical winner claim.',
                'Fixed active entity universe; prior observations available during rolling validation; no fixed-origin multi-period accuracy claim.',
                'Provider and original catalog promotion remain pending.']))

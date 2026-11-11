"""Single source fit subgraph, reusable for local residual/direct and global fits."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_nasa_first_fit_graph(*, residual):
    if type(residual) is not bool:
        raise ValueError('Explicit residual/direct mode required')
    def p(name,typ,contract):return IOSpec(name=name,type_desc=typ,constraints=contract)
    table = lambda name:p(name,'pd.DataFrame','Named materialized feature rows; preserve source column order and row identity.')
    targets = lambda name:p(name,'pd.Series','Finite row-aligned target values in minutes, after explicit offset subtraction.')
    split_inputs = [table('frame'),p('targets','np.ndarray','Aligned raw target values in minutes.'),
        p('times','pd.DatetimeIndex','Aligned naive timestamps.'),p('eligible','np.ndarray','Aligned boolean eligibility mask.'),
        p('cutoff','object','Declared naive cutoff; equality belongs to validation.'),
        p('offsets','np.ndarray','Aligned target offsets in minutes; baseline for residual, zero for direct.'),
        p('categorical_columns','list','Explicit ordered categorical feature names.')]
    split_outputs = [table('x_train'),targets('y_train'),table('x_valid'),targets('y_valid'),
        p('categorical_indices','list','Unique categorical indices into the ordered feature schema.')]
    policies = [p('parameters','dict','Pinned source CatBoost parameters; CPU execution and file/thread restrictions enforced by fit.'),
                p('early_stopping_rounds','int','Positive validation stopping patience; source value 60.')]
    def n(identity,runtime,inputs,outputs):
        return AlgorithmicNode(node_id=identity,name=identity,description='Explicit single native regression fit',concept_type='custom',
            status=NodeStatus.ATOMIC,matched_primitive=runtime,inputs=inputs,outputs=outputs)
    mode = 'residual_inputs' if residual else 'direct_inputs'
    nodes = [n('inputs','sciona.atoms.ml.domain_adapters.first_place_training.'+mode,
        [table('table'),p('target_name','str','Target column name.'),p('end_train','str','Explicit training/validation cutoff.')],split_inputs+policies),
        n('split','sciona.atoms.ml.model_selection.named_regression.temporal_split',split_inputs,split_outputs),
        n('fit','sciona.atoms.ml.model_selection.named_regression.fit',split_outputs+policies,
            [p('model','object','Fitted single-thread CPU native model; private runtime state.')])]
    edges=[]
    for source,target,ports in [('inputs','split',split_inputs),('split','fit',split_outputs),('inputs','fit',policies)]:
        edges.extend(DependencyEdge(source_id=source,target_id=target,output_name=port.name,input_name=port.name,
            source_type=port.type_desc,target_type=port.type_desc) for port in ports)
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',residual=residual,
        scope='Source fit input preparation and one bounded native regressor fit.',
        exclusions=['Single-candidate score acceptance and residual feature importance', '21-fit population assembly', 'Raw feature extraction']))

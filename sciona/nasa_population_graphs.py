"""Compile explicit population-count variants from approved lifecycle graphs.

The fixed-graph runner needs the active count at selection time. Selection does
not fit or predict: it expands the same visible operations once per population.
No airport identifiers or learned values are embedded in the compiled graph.
"""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.nasa_lifecycle_graphs import build_nasa_training_graph, build_nasa_inference_graph

ROUTING='sciona.atoms.ml.pipeline.keyed_routing.'
DOMAIN='sciona.atoms.ml.domain_adapters.airport_population.'


def select_population_graph(kind,payload):
    """Validate routing inputs and specialize by active count without model calls.

    This is compilation/selection, not an atom that executes a nested graph.
    Callers pass the unchanged payload to the selected visible graph.
    """
    from sciona.atoms.ml.domain_adapters.airport_population import training_jobs,inference_jobs
    if kind=='training':
        jobs=training_jobs(payload['training_records'],payload['airports'],payload['maximum_errors'])
    elif kind=='inference':
        jobs,_=inference_jobs(payload['prediction_records'],payload['model_states'])
    else:
        raise ValueError('Training or inference graph required')
    return build_population_graph(kind,len(jobs))


def _node(identity, function, inputs, outputs):
    return AlgorithmicNode(node_id=identity,name=function.rsplit('.',1)[-1].replace('_',' '),
        description='Explicit population routing; no hidden graph or model execution.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=function,
        inputs=[IOSpec(name=name,type_desc=typ,constraints=contract) for name,typ,contract in inputs],
        outputs=[IOSpec(name=name,type_desc=typ,constraints=contract) for name,typ,contract in outputs])


def build_population_graph(kind, population_count):
    if kind not in ('training','inference'):
        raise ValueError('Training or inference graph required')
    if type(population_count) is not int or population_count<1:
        raise ValueError('Positive active population count required')
    training=kind=='training'
    base=(build_nasa_training_graph if training else build_nasa_inference_graph)()
    contract='Insertion-ordered mapping with unique nonempty string keys; values are private borrowed runtime objects.'
    key_contract='Nonempty population key preserved from input partition to output; no implicit domain value.'
    nodes=[];edges=[]
    if training:
        inputs=[('training_records','dict','Caller-owned labeled query/event tables under the approved airport adapter contract.'),
                ('airports','tuple','Unique explicit four-character airport slots; every slot must have labeled query rows.'),
                ('maximum_errors','dict','Exactly one finite nonnegative minute residual cutoff per declared airport.')]
        outputs=[('jobs','dict',contract)]
    else:
        inputs=[('prediction_records','dict','Caller-owned nonempty query/event tables; unknown model populations are rejected.'),
                ('model_states','dict','Complete validated private model states keyed by their own airport; unqueried states may be present.')]
        outputs=[('jobs','dict',contract),('identities','DataFrame','Unique query identities retained in original input order.')]
    nodes.append(_node('partition',DOMAIN+kind+'_jobs',inputs,outputs))

    def connect(source,output,target,name):
        source_node=next(node for node in nodes if node.node_id==source)
        target_node=next(node for node in nodes if node.node_id==target)
        source_port=next(port for port in source_node.outputs if port.name==output)
        target_port=next(port for port in target_node.inputs if port.name==name)
        edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=name,
            source_type=source_port.type_desc,target_type=target_port.type_desc))

    boundary_names={'training_records','airport','maximum_error'} if training else {'model_state','prediction_records'}
    connected={(edge.target_id,edge.input_name) for edge in base.edges}
    boundary={port.name:port.model_dump() for node in base.nodes for port in node.inputs
              if (node.node_id,port.name) not in connected and port.name in boundary_names}
    previous='partition';previous_output='jobs';collected=None
    for index in range(population_count):
        prefix=f'p{index:02d}_'
        pop=prefix+'pop';slot=prefix+'slot';collect=prefix+'collect'
        nodes.append(_node(pop,ROUTING+'pop_first',[('mapping','dict',contract)],
            [('key','str',key_contract),('value','object','Borrowed mapping value; validated by the receiving adapter.'),('remaining','dict',contract)]))
        connect(previous,previous_output,pop,'mapping')
        slot_node=_node(slot,DOMAIN+kind+'_slot',[('job','dict','Explicit partitioned job validated by its lifecycle boundary.')],[])
        nodes.append(slot_node)
        # Provider tuples have a fixed ordinal contract.
        names=['training_records','airport','maximum_error'] if training else ['model_state','prediction_records']
        slot_node.outputs=[IOSpec(**boundary[name]) for name in names]
        connect(pop,'value',slot,'job')
        for original in base.nodes:
            node=original.model_copy(deep=True)
            node.node_id=prefix+original.node_id
            nodes.append(node)
        for original in base.edges:
            edge=original.model_copy(deep=True)
            edge.source_id=prefix+original.source_id;edge.target_id=prefix+original.target_id
            edges.append(edge)
        for node in base.nodes:
            for port in node.inputs:
                if (node.node_id,port.name) not in connected and port.name in boundary_names:
                    connect(slot,port.name,prefix+node.node_id,port.name)
        collector_inputs=([('mapping','dict',contract)] if collected else [])+[
            ('key','str',key_contract),('value','object','One completed branch result, borrowed without mutation.')]
        nodes.append(_node(collect,ROUTING+('insert_unique' if collected else 'singleton'),
            collector_inputs,[('mapping','dict',contract)]))
        if collected:connect(collected,'mapping',collect,'mapping')
        connect(pop,'key',collect,'key')
        connect(prefix+('learned_state' if training else 'domain_output'),
                'model_state' if training else 'prediction_table',collect,'value')
        previous=pop;previous_output='remaining';collected=collect
    nodes.append(_node('exhausted',ROUTING+'require_exhausted',
        [('remaining','dict',contract),('result','dict',contract)],[('result','dict',contract)]))
    connect(previous,'remaining','exhausted','remaining');connect(collected,'mapping','exhausted','result')
    if not training:
        nodes.append(_node('population_output',DOMAIN+'collect_predictions',
            [('identities','DataFrame','Unique original query identities in input order.'),('slot_predictions','dict',contract)],
            [('prediction_table','DataFrame','Exactly one int32 minute prediction per query, restored to original identity order.')]))
        connect('partition','identities','population_output','identities')
        connect('exhausted','result','population_output','slot_predictions')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(num_nodes=len(nodes),num_edges=len(edges),
        scope=f'Explicit corrected multi-airport {kind} graph for {population_count} active populations.',
        active_population_count=population_count,lifecycle_kind=kind,
        selection='Select count from explicit training slots or distinct queried airports before execution; exhaustion rejects mismatches.',
        exclusions=['Whole-graph static table propagation','Empirical competition performance','Catalog approval pending']))

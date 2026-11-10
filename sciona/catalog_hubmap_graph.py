"""Complete HuBMAP recipe with explicit reusable interfaces and source corrections."""
import json
from pathlib import Path
from sciona.hubmap_execution import build_hubmap_lifecycle_graph
from sciona.architect.models import IOSpec


def validate_root_contracts(graph, ports):
    """Require root contracts to match unconnected inputs and named leaf outputs."""
    nodes={n.node_id:n for n in graph.nodes}
    if len(nodes)!=len(graph.nodes):
        raise ValueError('Unique graph node identities required')
    incoming=set();outgoing=set()
    for edge in graph.edges:
        source=nodes.get(edge.source_id);target=nodes.get(edge.target_id)
        if (source is None or target is None or
                edge.output_name not in {p.name for p in source.outputs} or
                edge.input_name not in {p.name for p in target.inputs}):
            raise ValueError('Valid edge endpoints required')
        key=(edge.target_id,edge.input_name)
        if key in incoming:
            raise ValueError('One upstream binding per input required')
        incoming.add(key);outgoing.add((edge.source_id,edge.output_name))
    def contract(port):
        return {k:getattr(port,k) for k in ('name','type_desc','constraints','required','default_value_repr')}
    inputs={};leaves={}
    for node in graph.nodes:
        for port in node.inputs:
            if (node.node_id,port.name) not in incoming:
                value=contract(port)
                if port.name in inputs and inputs[port.name]!=value:
                    raise ValueError('Shared root input contracts disagree')
                inputs[port.name]=value
        for port in node.outputs:
            if (node.node_id,port.name) not in outgoing:
                leaves[(node.node_id,port.name)]=contract(port)
    sources=graph.metadata.get('catalog_output_sources',{})
    outputs={};selected=set()
    for name,source in sources.items():
        key=(source.get('node_id'),source.get('output_name'))
        if key not in leaves or key in selected:
            raise ValueError('Root outputs must select each leaf exactly once')
        selected.add(key);outputs[name]=dict(leaves[key],name=name)
    if selected!=set(leaves):
        raise ValueError('Every leaf output needs a root contract')
    expected=[]
    for direction,values in [('input',inputs),('output',outputs)]:
        expected.extend(dict(value,direction=direction,ordinal=i) for i,value in enumerate(values.values()))
    if sorted(ports,key=lambda p:(p['direction'],p['ordinal']))!=expected:
        raise ValueError('Root contracts differ from graph boundary')


def build_catalog_hubmap_graph():
    root=Path(__file__).resolve().parents[1]
    contracts=json.loads((root/'docs/reviews/hubmap_provider_contracts.json').read_text())
    scope=json.loads((root/'docs/reviews/hubmap_original_scope.json').read_text())
    if contracts.get('passed') is not True or scope.get('passed') is not True:
        raise ValueError('Passing source and provider qualifications required')
    graph=build_hubmap_lifecycle_graph()
    providers={r['runtime_fqdn']:r for r in contracts['providers']}
    if len(graph.nodes)!=6 or len(graph.edges)!=5 or set(providers)!={n.matched_primitive for n in graph.nodes}:
        raise ValueError('Complete six-node lifecycle and five reviewed providers required')
    for node in graph.nodes:
        record=providers[node.matched_primitive]
        for direction,field in [('input','inputs'),('output','outputs')]:
            ports=sorted((p for p in record['ports'] if p['direction']==direction),key=lambda p:p['ordinal'])
            if [(p.name,p.type_desc) for p in getattr(node,field)]!=[(p['name'],p['type_desc']) for p in ports]:
                raise ValueError('Reviewed callable ports differ')
            setattr(node,field,[IOSpec(**{k:p[k] for k in ('name','type_desc','constraints','required','default_value_repr')}) for p in ports])
        node.description=record['plain_description']
    graph.metadata.update(artifact_source='corrected_original_competition_execution',
        original_source_version_id=scope['source_version_id'],original_source_content_hash=scope['source_content_hash'],
        catalog_output_sources={
            'initial_metrics':dict(node_id='initial_training',output_name='initial_metrics'),
            'retraining_metrics':dict(node_id='retraining',output_name='retraining_metrics'),
            'predictions':dict(node_id='final_prediction',output_name='predictions')},
        source_corrections={r['stage_id']:r['realization']+' '+r['correction'] for r in scope['source_stage_review']},
        scope='Full grouped raster preparation, initial fold training, pseudo-mask generation, fresh retraining and final prediction. Caller justifies foreground meaning, physical scale, independent populations, pseudo-source isolation and explicit checkpoint selection.',
        reuse_decision=scope['reuse_decision'],limits=list(dict.fromkeys(scope['limitations']+contracts['limitations'])),
        qualification_status='Corrected stored original execution and publication remain pending.')
    validate_root_contracts(graph,contracts['derivative']['ports'])
    return graph

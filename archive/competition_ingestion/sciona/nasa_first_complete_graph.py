"""Complete first-place lifecycle with an explicit portable-state dependency.

Training and inference retain their qualified nodes. Source prediction accepts
one population per call; training produces the complete ten-population bank.
"""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import DependencyEdge
from sciona.nasa_first_state_graphs import (
    build_nasa_first_training_state_graph, build_nasa_first_state_inference_graph,
)
from sciona.services.execution_graph_codec import encode_execution_graph


def build_nasa_first_complete_graph():
    training=build_nasa_first_training_state_graph()
    inference=build_nasa_first_state_inference_graph()
    nodes=[];edges=[]
    for prefix,graph in [('training/',training),('inference/',inference)]:
        nodes.extend(node.model_copy(deep=True,update={'node_id':prefix+node.node_id}) for node in graph.nodes)
        edges.extend(edge.model_copy(deep=True,update={'source_id':prefix+edge.source_id,'target_id':prefix+edge.target_id}) for edge in graph.edges)
    edges.append(DependencyEdge(source_id='training/bind_state',target_id='inference/unbind_state',
        output_name='state',input_name='state',source_type='dict',target_type='dict'))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        scope='Corrected raw ten-population training, portable policy-bound state and guarded single-population prediction.',
        training_graph_sha256=encode_execution_graph(training)[0],
        inference_graph_sha256=encode_execution_graph(inference)[0],
        primary_branch=inference.metadata['primary_branch'],baseline_branch=inference.metadata['baseline_branch'],
        training_output_node='training/bind_state',training_output_port='state',
        final_node_id='inference/format',final_output_port='predictions',
        exclusions=['Empirical competition performance','Original trained weights','Historical backend identity',
                    'Static symbolic propagation','Source notebook and filesystem orchestration']))


def root_contracts(graph):
    """Expose only unwired inputs; retain every applicable explicit constraint."""
    connected={(edge.target_id,edge.input_name) for edge in graph.edges}
    roots={}
    for node in graph.nodes:
        for port in node.inputs:
            if (node.node_id,port.name) in connected:continue
            value=port.model_dump(mode='json')
            if port.name not in roots:
                roots[port.name]=value
            else:
                existing=roots[port.name]
                for key in ['type_desc','required','default_value_repr','dim_signature']:
                    if existing[key]!=value[key]:raise ValueError('Conflicting lifecycle root contract: '+port.name)
                if value['constraints'] not in existing['constraints']:
                    existing['constraints']+=' '+value['constraints']
    if 'state' in roots:raise ValueError('Training state must be internally connected')
    return list(roots.values())

import pytest

from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph
from sciona.nasa_population_graphs import build_population_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer.runner import get_topo_sorted_leaves


@pytest.mark.parametrize('count',range(1,11))
def test_composition_preserves_both_regions_and_only_adds_forward_state_handoff(count):
    graph=build_complete_workflow_graph(count)
    for kind,size in [('training',10),('inference',count)]:
        prefix=kind+'_';base=build_population_graph(kind,size)
        nodes=[];edges=[]
        for original in graph.nodes:
            if original.node_id.startswith(prefix):
                node=original.model_copy(deep=True);node.node_id=node.node_id[len(prefix):];nodes.append(node)
        for original in graph.edges:
            if original.source_id.startswith(prefix) and original.target_id.startswith(prefix):
                edge=original.model_copy(deep=True)
                edge.source_id=edge.source_id[len(prefix):];edge.target_id=edge.target_id[len(prefix):];edges.append(edge)
        assert nodes==base.nodes and edges==base.edges
    cross=[edge for edge in graph.edges if edge.source_id.split('_')[0]!=edge.target_id.split('_')[0]]
    assert len(cross)==1
    assert (cross[0].source_id,cross[0].output_name,cross[0].target_id,cross[0].input_name)==(
        'training_exhausted','result','inference_partition','model_states')
    connected={(edge.target_id,edge.input_name) for edge in graph.edges}
    roots={port.name for node in graph.nodes for port in node.inputs if (node.node_id,port.name) not in connected}
    assert roots=={'training_records','airports','maximum_errors','train_fraction','seed','threshold',
                   'prediction_feature_name','prediction_records'}
    assert all(port.name!='prediction_records' for node in graph.nodes if node.node_id.startswith('training_') for port in node.inputs)
    assert len(get_topo_sorted_leaves(graph.nodes,graph.edges))==len(graph.nodes)
    digest,nodes,edges=encode_execution_graph(graph)
    assert decode_execution_graph(nodes,edges,digest)==graph

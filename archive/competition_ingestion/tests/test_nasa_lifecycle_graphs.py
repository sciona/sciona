from sciona.nasa_lifecycle_graphs import build_nasa_training_graph,build_nasa_inference_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer.runner import get_topo_sorted_leaves


def roots(graph):
    connected={(edge.target_id,edge.input_name) for edge in graph.edges}
    return {port.name for node in graph.nodes for port in node.inputs if (node.node_id,port.name) not in connected}


def test_training_graph_has_no_query_population_dependency():
    graph=build_nasa_training_graph()
    assert roots(graph)=={'training_records','airport','train_fraction','seed','maximum_error','threshold','prediction_feature_name'}
    identifiers={node.node_id for node in graph.nodes}
    assert {'internal_fit','lower_fit','classifier_fit','offsets','domain_vocabulary','learned_state'}<=identifiers
    assert not any('query' in identity for identity in identifiers)
    assert len(get_topo_sorted_leaves(graph.nodes,graph.edges))==len(graph.nodes)


def test_inference_graph_requires_saved_state_and_has_no_fit_operations():
    graph=build_nasa_inference_graph()
    assert roots(graph)=={'model_state','prediction_records'}
    functions=[node.matched_primitive.rsplit('.',1)[-1] for node in graph.nodes]
    assert not any('fit' in function or function=='training_handoff' for function in functions)
    assert {'unpack_model_state','predict_regression_model','predict_binary_model','apply_float32_offsets','round_to_int32'}<=set(functions)
    assert len(get_topo_sorted_leaves(graph.nodes,graph.edges))==len(graph.nodes)


def test_each_lifecycle_graph_preserves_its_execution_envelope():
    for builder in [build_nasa_training_graph,build_nasa_inference_graph]:
        graph=builder()
        digest,nodes,edges=encode_execution_graph(graph)
        assert decode_execution_graph(nodes,edges,digest)==graph

import inspect

from sciona.atoms.dl import grapheme_execution  # noqa: F401: registers providers
from sciona.ghost.registry import REGISTRY
from sciona.grapheme_execution import build_grapheme_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph


def test_complete_graph_roundtrip_and_real_provider_signatures():
    graph = build_grapheme_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    assert decode_execution_graph(nodes, edges, digest) == graph
    assert len(graph.nodes) == 22 and len(graph.edges) == 32
    for node in graph.nodes:
        entry = REGISTRY[node.matched_primitive]
        names = [port.name for port in node.inputs]
        assert list(inspect.signature(entry['impl']).parameters) == names
        assert list(inspect.signature(entry['witness']).parameters) == names
    assert len({node.matched_primitive for node in graph.nodes}) == 14


def test_calibration_and_query_paths_are_separate_and_every_training_state_is_used():
    graph = build_grapheme_graph()
    nodes = {node.node_id: node for node in graph.nodes}
    targets = set()
    for edge in graph.edges:
        assert edge.output_name in {port.name for port in nodes[edge.source_id].outputs}
        assert edge.input_name in {port.name for port in nodes[edge.target_id].inputs}
        key = edge.target_id, edge.input_name
        assert key not in targets
        targets.add(key)
        if edge.source_id == 'inference_inputs':
            assert edge.output_name == ('validation_images' if edge.target_id.endswith('_validation') else 'query_images')
    for identity in ('train_ood', 'train_seen', 'train_font_a', 'train_font_b', 'train_generator_a', 'train_generator_b'):
        assert any(edge.source_id == identity for edge in graph.edges)
    assert {edge.source_id for edge in graph.edges if edge.target_id == 'calibration'} == {'scores_validation', 'ensemble_validation'}
    assert {edge.source_id for edge in graph.edges if edge.target_id == 'routing'} == {'scores_query', 'ensemble_query', 'calibration'}

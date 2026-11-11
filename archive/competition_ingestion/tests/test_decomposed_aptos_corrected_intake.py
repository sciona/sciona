"""Preserve the full source boundary and explicit cut dataflow in original intake."""
import pytest
from scripts import stage_aptos_corrected_intake as intake
from scripts.build_aptos_execution_graph import build_graph as reference_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph


def test_complete_source_inputs_and_stored_codec():
    graph = intake.build_graph()
    inputs, outputs = intake.boundaries(graph)
    reference = reference_graph()
    expected = reference.nodes[0].inputs + [p for p in reference.nodes[1].inputs if p.name != 'plan']
    assert inputs == [p.model_dump() for p in expected]
    assert len(inputs) == 15 and [p['name'] for p in outputs] == ['result']
    digest, nodes, edges = encode_execution_graph(graph)
    assert len(nodes) == 5 and len(edges) == 7
    assert decode_execution_graph(nodes, edges, digest) == graph
    assert graph.metadata['original_source_version_id'] == intake.ORIGINAL
    assert graph.metadata['original_source_content_hash'] == intake.ORIGINAL_HASH
    assert graph.metadata['qualified_reference_version_id'] == intake.PARENT


@pytest.mark.parametrize('fault', ['missing_input', 'missing_cut_edge', 'missing_output'])
def test_incomplete_boundary_rejected(fault):
    graph = intake.build_graph()
    if fault == 'missing_input':
        node = next(n for n in graph.nodes if n.node_id == 'train_scores')
        node.inputs = [p for p in node.inputs if p.name != 'images']
    elif fault == 'missing_cut_edge':
        graph.edges = [e for e in graph.edges if not (e.target_id == 'classify' and e.input_name == 'cut_points')]
    else:
        graph.metadata['catalog_output_sources'] = {}
    with pytest.raises(ValueError, match='Complete root boundary differs'):
        intake.boundaries(graph)

"""Execute synthetic cases using the exact approved catalog port contracts."""
import copy

import pytest

from sciona.catalog_text_pair_complete_graph import build_catalog_text_pair_graph
from sciona.text_pair_complete_contract import execute, prepare
from tests.test_text_pair_complete_graph import field_payload
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('change', ['baseline', 'distinct_titles', 'repeated_query'])
def test_catalog_graph_preserves_complete_recipe(change, tmp_path, monkeypatch):
    p = field_payload()
    if change == 'distinct_titles':
        p['query_pairs'][0][0]['title'] = 'purple cloud'
    if change == 'repeated_query':
        p['query_pairs'].append(copy.deepcopy(p['query_pairs'][0]))
    before = copy.deepcopy(p)
    graph = build_catalog_text_pair_graph()
    assert all(port.constraints for node in graph.nodes for port in node.inputs+node.outputs)
    captured = run(graph, {'payload': p}, tmp_path, monkeypatch)
    assert captured['format_rows', 'out_result'] == execute(prepare(p))
    assert p == before
    for edge in graph.edges:
        assert captured[edge.source_id, 'out_'+edge.output_name] is captured[edge.target_id, 'in_'+edge.input_name]

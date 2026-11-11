"""Synthetic execution using exact approved tabular catalog contracts."""
import copy
import pytest
from sciona.catalog_tabular_complete_graph import build_catalog_tabular_graph
from sciona.tabular_complete_contract import execute, prepare
from tests.test_tabular_complete_contract import payload
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('change', ['baseline', 'unknown_query', 'repeated_query'])
def test_catalog_graph_preserves_complete_recipe(change, tmp_path, monkeypatch):
    p = payload()
    if change == 'unknown_query': p['query']['categorical'][0] = ['unseen words']
    if change == 'repeated_query':
        for key in ('numeric', 'categorical', 'groups'):
            p['query'][key].append(copy.deepcopy(p['query'][key][0]))
    before = copy.deepcopy(p)
    graph = build_catalog_tabular_graph()
    assert len(graph.nodes) == 4 and len(graph.edges) == 3
    assert all(port.constraints for node in graph.nodes for port in node.inputs+node.outputs)
    captured = run(graph, {'payload':p}, tmp_path, monkeypatch)
    assert captured['predict', 'out_result'] == execute(prepare(p))
    assert p == before
    for edge in graph.edges:
        assert captured[edge.source_id, 'out_'+edge.output_name] is captured[edge.target_id, 'in_'+edge.input_name]

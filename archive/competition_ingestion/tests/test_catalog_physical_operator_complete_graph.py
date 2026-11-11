"""Full synthetic physical lifecycle through uncached serialized native nodes."""
import copy
import numpy as np
import pytest
from scripts.physical_operator_synthetic import payload
from sciona.catalog_physical_operator_complete_graph import build_catalog_physical_operator_graph
from sciona.physical_operator_contract import prepare, execute
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('change', ['baseline', 'repeated_query', 'no_smoothing'])
def test_catalog_complete_lifecycle(change, tmp_path, monkeypatch):
    p = payload()
    if change == 'repeated_query':
        for field in ('states', 'diffusivity', 'elapsed', 'groups'):
            p['query'][field].append(copy.deepcopy(p['query'][field][0]))
    if change == 'no_smoothing':
        p['controls']['smoothing'] = 0
    original = copy.deepcopy(p)
    graph = build_catalog_physical_operator_graph()
    assert all(p.constraints for n in graph.nodes for p in n.inputs+n.outputs)
    captured = run(graph, {'payload':p}, tmp_path, monkeypatch)
    assert len(graph.nodes) == 5 and len(graph.edges) == 4
    for edge in graph.edges:
        assert captured[edge.source_id, 'out_'+edge.output_name] is captured[edge.target_id, 'in_'+edge.input_name]
    expected = execute(prepare(p))
    actual = captured['finish', 'out_result']
    assert set(actual) == set(expected)
    for field in actual:
        np.testing.assert_allclose(actual[field], expected[field], rtol=1e-12, atol=1e-12)
    assert p == original

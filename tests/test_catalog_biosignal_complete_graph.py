from copy import deepcopy
import numpy as np
import pytest
from sciona.catalog_biosignal_complete_graph import build_catalog_biosignal_graph
from sciona.biosignal_sequence_contract import prepare, execute
from tests.test_biosignal_complete_stages import case
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('change', ['baseline', 'no_mask', 'single_batch'])
def test_uncached_lifecycle(change, tmp_path, monkeypatch):
    p = case(change)
    original = deepcopy(p)
    graph = build_catalog_biosignal_graph()
    captured = run(graph, {'payload': p}, tmp_path, monkeypatch)
    for edge in graph.edges:
        assert captured[edge.source_id, 'out_' + edge.output_name] is captured[edge.target_id, 'in_' + edge.input_name]
    result = captured['predict', 'out_result']
    expected = execute(prepare(p))
    assert result.keys() == expected.keys()
    for key in result:
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-12, atol=1e-12)
    assert p == original

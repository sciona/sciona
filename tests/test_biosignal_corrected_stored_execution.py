from copy import deepcopy
import numpy as np
import pytest
from scripts import stage_biosignal_corrected_intake as intake
from sciona.biosignal_sequence_contract import prepare, execute
from tests.test_biosignal_complete_stages import case
from tests.test_text_pair_provider_graphs import run


@pytest.fixture(scope='module')
def stored_graph():
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        return intake.check_candidate(db, plan, providers)


@pytest.mark.parametrize('change', ['baseline', 'no_mask', 'single_batch'])
def test_uncached_lifecycle(stored_graph, change, tmp_path, monkeypatch):
    p = case(change)
    original = deepcopy(p)
    graph = stored_graph
    captured = run(graph, {'payload': p}, tmp_path, monkeypatch)
    for edge in graph.edges:
        assert captured[edge.source_id, 'out_' + edge.output_name] is captured[edge.target_id, 'in_' + edge.input_name]
    result = captured['predict', 'out_result']
    expected = execute(prepare(p))
    assert result.keys() == expected.keys()
    for key in result:
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-12, atol=1e-12)
    assert p == original

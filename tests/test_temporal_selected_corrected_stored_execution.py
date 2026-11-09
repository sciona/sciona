from copy import deepcopy
import pytest
from scripts.temporal_sparse_synthetic import payload
from sciona.temporal_sparse_selected_execution import prepare, execute
from scripts import stage_temporal_selected_corrected_intake as intake
from tests.test_text_pair_provider_graphs import run


@pytest.fixture(scope='module')
def stored_graph():
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        return intake.check_candidate(db, plan, providers)


@pytest.mark.parametrize('selection', [dict(max_epochs=1, patience=2, min_delta=0.),
    dict(max_epochs=4, patience=2, min_delta=0.), dict(max_epochs=8, patience=2, min_delta=1e6)])
def test_native_selected_lifecycle(stored_graph, selection, tmp_path, monkeypatch):
    config = payload(tmp_path)
    config['selection'] = selection
    original = deepcopy(config)
    expected = execute(prepare(config))
    graph = stored_graph
    actual = run(graph, {'payload': config}, tmp_path, monkeypatch)
    assert actual['prepare', 'out_result'] is actual['execute', 'in_prepared']
    assert actual['execute', 'out_result'] == expected
    assert config == original
    assert all(p.constraints for node in graph.nodes for p in node.inputs + node.outputs)

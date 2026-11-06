"""Synthetic native graph parity and execution-state routing."""
import copy
import pytest

from sciona.tabular_complete_contract import execute, prepare
from sciona.tabular_complete_graph import build_complete_tabular_graph, STAGES
from tests.test_tabular_complete_contract import payload
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('change', ['baseline', 'unknown_query', 'repeated_query'])
def test_native_graph_parity_and_state_identity(change, tmp_path, monkeypatch):
    p = payload()
    if change == 'unknown_query': p['query']['categorical'][0] = ['unseen words']
    if change == 'repeated_query':
        for key in ('numeric', 'categorical', 'groups'):
            p['query'][key].append(copy.deepcopy(p['query'][key][0]))
    before = copy.deepcopy(p)
    expected = execute(prepare(p))
    captured = run(build_complete_tabular_graph(), {'payload': p}, tmp_path, monkeypatch)
    assert captured['predict', 'out_result'] == expected
    assert p == before
    for source, target, input_name in zip(STAGES, STAGES[1:], ('prepared', 'fitted', 'calibrated')):
        assert captured[source, 'out_result'] is captured[target, 'in_'+input_name]
    fitted = captured['fit_stack', 'out_result']
    calibrated = captured['calibrate', 'out_result']
    assert calibrated.fitted_state is fitted
    assert fitted.fitted.oof_predictions.shape == (12, 2)
    assert fitted.fitted.base.trees.n_jobs == 1

"""Complete synthetic registered lifecycles, exact state routing and source parity."""
from copy import deepcopy
import numpy as np
import pytest
from scripts.image_tabular_synthetic import payload
from sciona.image_tabular_complete_graph import build_complete_image_tabular_graph
from sciona.image_tabular_contract import prepare,execute
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('change',['baseline','no_dropout','no_smoothing'])
def test_uncached_full_lifecycle(change,tmp_path,monkeypatch):
    p=payload()
    if change=='no_dropout':p['controls']['dropout']=0.
    if change=='no_smoothing':p['controls']['smoothing']=0.
    before=deepcopy(p)
    graph=build_complete_image_tabular_graph()
    captured=run(graph,{'payload':p},tmp_path,monkeypatch)
    for edge in graph.edges:
        assert captured[edge.source_id,'out_'+edge.output_name] is captured[edge.target_id,'in_'+edge.input_name]
    result=captured['predict','out_result'];expected=execute(prepare(p))
    assert result.keys()==expected.keys()
    for key in result:np.testing.assert_allclose(result[key],expected[key],rtol=1e-12,atol=1e-12)
    assert p==before

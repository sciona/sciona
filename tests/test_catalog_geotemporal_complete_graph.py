"""Uncached registered five-stage execution with exact object routing."""
from copy import deepcopy
import numpy as np
import pytest
from sciona.catalog_geotemporal_complete_graph import build_catalog_geotemporal_graph
from sciona.geotemporal_contract import prepare,execute
from tests.test_geotemporal_stages import changed_payload
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('change',['baseline','no_smoothing','separate_query_times'])
def test_full_native_lifecycle(change,tmp_path,monkeypatch):
    p=changed_payload(change);original=deepcopy(p);graph=build_catalog_geotemporal_graph()
    assert all(port.constraints for node in graph.nodes for port in node.inputs+node.outputs)
    captured=run(graph,{'payload':p},tmp_path,monkeypatch)
    assert len(graph.nodes)==5 and len(graph.edges)==4
    for edge in graph.edges:
        assert captured[edge.source_id,'out_'+edge.output_name] is captured[edge.target_id,'in_'+edge.input_name]
    actual=captured['finish','out_result'];expected=execute(prepare(p))
    assert actual.keys()==expected.keys()
    for key in actual:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
    assert p==original

"""Native execution with explicit reviewed usage contracts and retained bindings."""
from copy import deepcopy
import pytest

from sciona.catalog_resource_route_graph import build_catalog_resource_route_graph
from sciona.resource_route_planning import plan
from scripts.resource_route_synthetic import payload
from tests.test_text_pair_provider_graphs import run


@pytest.fixture(scope='module')
def graph():
    return build_catalog_resource_route_graph()


@pytest.mark.parametrize('case',['optimal','infeasible','search_limit','identity'])
def test_qualified_route_contracts(graph,case,tmp_path,monkeypatch):
    p=payload()
    if case=='infeasible':p['blocked_edges']=['detour']
    if case=='search_limit':p['max_expansions']=1
    if case=='identity':p['goal']=p['start']
    original=deepcopy(p)
    assert all(port.constraints for node in graph.nodes for port in node.inputs+node.outputs)
    captured=run(graph,{'payload':p},tmp_path,monkeypatch)
    for edge in graph.edges:
        assert captured[edge.source_id,'out_'+edge.output_name] is captured[edge.target_id,'in_'+edge.input_name]
    result=captured['select','out_result']
    assert result==plan(p)
    assert result['status']==('optimal' if case=='identity' else case)
    assert p==original

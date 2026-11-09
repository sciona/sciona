"""Execute the retrieved original candidate with all native stages uncached."""
from copy import deepcopy
import pytest

from scripts import stage_resource_route_corrected_intake as intake
from sciona.resource_route_planning import plan
from scripts.resource_route_synthetic import payload
from tests.test_text_pair_provider_graphs import run


@pytest.fixture(scope='module')
def graph():
    plan,providers=intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        return intake.check_candidate(db,plan,providers)


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

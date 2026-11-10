"""Synthetic full solver execution; no competition task records."""
from copy import deepcopy
import pytest
from sciona.catalog_arc_graph import build_catalog_arc_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('count',[1,2])
def test_full_catalog_solver(count,tmp_path,monkeypatch):
    # Local reviewed bundle is deployment configuration supplied by the validator.
    payload={'version':1,'task':{'train':[
        {'input':[[0,1],[1,0]],'output':[[0,1],[1,0]]},
        {'input':[[2,0],[0,2]],'output':[[2,0],[0,2]]}],
        'test':[{'input':[[0,3],[3,0]]},{'input':[[4,0],[0,4]]}][:count]},
        'budget':{'seconds':30,'memory_mib':1024}}
    before=deepcopy(payload)
    digest,nodes,edges=encode_execution_graph(build_catalog_arc_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    captured=run(graph,{'payload':payload},tmp_path,monkeypatch)
    assert captured['prepare','out_prepared'] is captured['execute','in_prepared']
    result=captured['execute','out_result']
    assert result['completed_all_runs'] and result['missing_test_indices']==[]
    assert result['search_arguments']==[3,23,33,4]
    assert len(result['attempts'])==4*count
    assert all(a['status']=='success' for a in result['attempts'])
    for item,candidates in zip(payload['task']['test'],result['predictions'],strict=True):
        assert item['input'] in [c['grid'] for c in candidates]
        assert 1<=len(candidates)<=3
    assert payload==before

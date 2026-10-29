"""Execute the decomposed mass stage through the production CDG runner."""
import asyncio
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sciona.flavours_mass_execution import build_mass_regression_graph
from sciona.flavours_mass_correction import correct_mass
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


@pytest.mark.parametrize('width,seed',[(3,812),(5,917)])
def test_full_mass_graph_roundtrip_execution_matches_reference(tmp_path,monkeypatch,width,seed):
    graph=build_mass_regression_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg(dict(
        cdg_nodes=[dict(n,version_id='synthetic-validation') for n in nodes],
        cdg_edges=[dict(e,version_id='synthetic-validation') for e in edges]),
        version_id='synthetic-validation',content_hash=digest,require_execution_envelope=True)
    assert restored==graph
    assert len(graph.nodes)==281
    bound={(e.target_id,e.input_name) for e in graph.edges}
    roots={(n.node_id,p.name) for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in bound}
    assert {name for _,name in roots}=={'features','query_features','targets','feature_names'}
    assert len(bound)==len(graph.edges)
    rng=np.random.default_rng(seed)
    train=rng.normal(size=(63,width));query=rng.normal(size=(11,width))
    target=2*train[:,0]+train[:,1]**2
    train[::13,-1]=np.nan;query[::5,-1]=np.nan
    inputs=dict(features=train,query_features=query,targets=target,
                feature_names=[f'feature_{i}' for i in range(width)])
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    with threadpool_limits(limits=1):
        expected=correct_mass(train,target,query)
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-mass-stage','check').execute(inputs,cdg=restored))
    assert result['status']=='completed' and len(result['trace'])==281
    assert all(not item['cached'] for item in result['trace'])
    actual_held=np.load(tmp_path/'check/held_mean/out_predictions.npy')
    actual_query=np.load(tmp_path/'check/query_mean/out_predictions.npy')
    np.testing.assert_array_equal(actual_held,expected['oof_mass'])
    np.testing.assert_array_equal(actual_query,expected['query_mass'])
    assert np.ptp(actual_query)>0
    for repeat in range(7):
        for fold in range(5):
            node=f'r{repeat}_f{fold}_fit'
            # Actual-round outputs must be retained even though no downstream
            # arithmetic node consumes them.
            assert json.loads((tmp_path/'check'/node/'out_actual_rounds.json').read_text())['value']==2800
            fit_mask=np.load(tmp_path/'check'/node/'in_selection.npy')
            held_mask=np.load(tmp_path/'check'/f'r{repeat}_f{fold}_held_contribution'/'in_selection.npy')
            assert np.all(fit_mask ^ held_mask)

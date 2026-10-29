"""Full-budget neural CDG execution and explicit diagnostic boundaries."""
import asyncio
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sciona.flavours_neural import train_neural
from sciona.flavours_neural_execution import build_neural_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from sciona.atoms.ml.numpy_adagrad_binary import fit_masked_adagrad_binary
from sciona.atoms.ml.neural_evaluation import masked_neural_loss,masked_neural_predictions
from sciona.atoms.ml.model_selection.scaled_prediction_contributions import divided_prediction_contribution,finalize_prediction_sum


def test_full_neural_graph_matches_reference(tmp_path,monkeypatch):
    graph=build_neural_graph();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg(dict(cdg_nodes=[dict(n,version_id='check') for n in nodes],
        cdg_edges=[dict(e,version_id='check') for e in edges]),version_id='check',content_hash=digest,require_execution_envelope=True)
    assert restored==graph and len(nodes)==426
    bound={(e.target_id,e.input_name) for e in graph.edges}
    assert len(bound)==len(edges)
    assert {p.name for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in bound}=={'features','targets','query_features','feature_names'}
    rng=np.random.default_rng(641);x=rng.normal(size=(600,3));q=rng.normal(size=(17,3))
    y=np.tile(np.array([0,1],dtype=np.int64),300);rng.shuffle(y);x[:,0]+=y*.7
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    with threadpool_limits(limits=1):
        expected=train_neural(x,y,q)
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-neural-stage','check').execute(
            dict(features=x,targets=y,query_features=q,feature_names=['a','b','c']),cdg=restored))
    assert result['status']=='completed' and len(result['trace'])==426
    assert all(not item['cached'] for item in result['trace'])
    np.testing.assert_array_equal(np.load(tmp_path/'check/held_sum/out_predictions.npy'),expected['oof'])
    np.testing.assert_array_equal(np.load(tmp_path/'check/query_sum/out_predictions.npy'),expected['query'])
    assert np.ptp(expected['oof'])>0 and np.ptp(expected['query'])>0
    for record in expected['fits']:
        prefix=f"f{record['fold']}_s{record['seed']}"
        for output,key in [('epochs','epochs'),('updates','optimizer_updates')]:
            value=json.loads((tmp_path/'check'/f'{prefix}_fit'/f'out_{output}.json').read_text())['value']
            assert value==record[key]
        loss=json.loads((tmp_path/'check'/f'{prefix}_validation'/'out_loss.json').read_text())['value']
        assert loss==record['validation_loss']
    for fold in range(5):
        outer=np.load(tmp_path/'check'/f'f{fold}_scaler'/'in_selection.npy')
        inner=np.load(tmp_path/'check'/f'f{fold}_s0_fit'/'in_selection.npy')
        validation=np.load(tmp_path/'check'/f'f{fold}_s0_validation'/'in_selection.npy')
        held=np.load(tmp_path/'check'/f'f{fold}_s0_held_prediction'/'in_selection.npy')
        np.testing.assert_array_equal(inner|validation,outer)
        assert not np.any(inner&validation) and np.all(outer^held)


def test_selected_neural_diagnostics_ignore_excluded_values_and_do_not_mutate_state():
    x=np.arange(24,dtype=float).reshape(8,3)/24;y=np.arange(8,dtype=np.int64)%2;names=['a','b','c']
    config=dict(hidden_units=3,epochs=2,batch_size=4,seed=1,learning_rate=.01,epsilon=1e-6)
    state,_,_=fit_masked_adagrad_binary(x,y,names,np.ones(8,dtype=bool),config)
    before=json.dumps(state);mask=np.arange(8)%2==0
    p=masked_neural_predictions(x,names,state,mask);loss=masked_neural_loss(x,y,names,state,mask)
    changed=x.copy();changed[~mask]=np.inf;labels=y.copy();labels[~mask]=-77
    np.testing.assert_array_equal(masked_neural_predictions(changed,names,state,mask),p)
    assert masked_neural_loss(changed,labels,names,state,mask)==loss and json.dumps(state)==before
    assert np.all(p[~mask]==0)
    labels[mask]=-1
    with pytest.raises(ValueError):masked_neural_loss(x,labels,names,state,mask)


def test_divided_contributions_preserve_source_order_and_require_exact_coverage():
    values=np.array([.7,np.nan]);mask=np.array([True,False])
    total,counts=divided_prediction_contribution(values,mask,10.)
    np.testing.assert_array_equal(total,[.7/10,0.]);np.testing.assert_array_equal(counts,[1,0])
    with pytest.raises(ValueError):finalize_prediction_sum(total,counts,1)
    for divisor in [0.,-1.,np.inf,True]:
        with pytest.raises(ValueError):divided_prediction_contribution(values,mask,divisor)
    source=np.array([.7/10]);actual=finalize_prediction_sum(source,np.array([10],dtype=np.int64),10)
    np.testing.assert_array_equal(actual,source);assert not np.shares_memory(actual,source)

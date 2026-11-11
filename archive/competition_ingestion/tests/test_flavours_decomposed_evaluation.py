"""Execute all training and diagnostic nodes against the complete reference."""
import asyncio
import numpy as np
from threadpoolctl import threadpool_limits
from sciona.flavours_evaluated_execution import execute_evaluated
from sciona.flavours_decomposed_evaluation import build_evaluated_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


def test_complete_evaluated_graph_matches_reference(tmp_path,monkeypatch):
    graph=build_evaluated_graph();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg(dict(cdg_nodes=[dict(n,version_id='check') for n in nodes],
        cdg_edges=[dict(e,version_id='check') for e in edges]),version_id='check',content_hash=digest,require_execution_envelope=True)
    assert restored==graph and len(nodes)==841
    bound={(e.target_id,e.input_name) for e in graph.edges}
    assert len(bound)==len(edges)
    assert {p.name for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in bound}=={
        'training_inputs','labels','mass','query_inputs','controls','excluded_column'}
    for metric in ('agreement_metric','correlation_metric','discrimination_metric'):
        assert all(e.target_id=='report' for e in graph.edges if e.source_id==metric)
    rng=np.random.default_rng(624)
    def physical(rows):
        base=rng.normal(size=(rows,46));p=rng.uniform(1,8,(rows,3));pt=p*rng.uniform(.1,.8,(rows,3))
        return [base,p,pt,rng.uniform(.2,3,rows),rng.uniform(1,5,rows),rng.uniform(.1,2,rows)]
    training=physical(600);query=physical(17)
    labels=np.tile(np.array([0,1],dtype=np.int64),300);rng.shuffle(labels);training[0][:,0]+=labels*.8
    controls=dict(agreement_a=physical(25),agreement_b=physical(27),correlation=physical(300),
        weights_a=rng.uniform(.5,2,25),weights_b=rng.uniform(.5,2,27),correlation_mass=rng.uniform(1,7,300),quality=np.ones(600))
    inputs=dict(training_inputs=training,labels=labels,mass=3+training[0][:,1]**2+.5*training[0][:,2],
        query_inputs=query,controls=controls,excluded_column=17)
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    captured={};original_save=runner.save_intermediate_value
    def save(directory,node_id,name,value):
        if node_id=='report' and name=='out_result':captured['result']=value
        return original_save(directory,node_id,name,value)
    monkeypatch.setattr(runner,'save_intermediate_value',save)
    with threadpool_limits(limits=1):
        expected=execute_evaluated(**inputs)
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-evaluated-pipeline','check').execute(inputs,cdg=restored))
    assert result['status']=='completed' and len(result['trace'])==841
    assert all(not item['cached'] for item in result['trace'])
    actual=captured['result']
    np.testing.assert_array_equal(actual['query_scores'],expected['query_scores'])
    assert actual['evaluation']==expected['evaluation']
    assert actual['population_sizes']==expected['population_sizes']==(17,25,27,300)
    assert np.ptp(actual['query_scores'])>0
    np.testing.assert_array_equal(np.load(tmp_path/'check/query_blend/out_scores.npy'),expected['execution']['query_scores'])
    np.testing.assert_array_equal(np.load(tmp_path/'check/oof_blend/out_scores.npy'),expected['execution']['oof_scores'])
    fit_nodes=[n for n in graph.nodes if n.matched_primitive.endswith(('.fit_masked_configured_regression','.fit_masked_configured_binary',
        '.fit_configured_binary_model','.fit_masked_adagrad_binary'))]
    assert len(fit_nodes)==115

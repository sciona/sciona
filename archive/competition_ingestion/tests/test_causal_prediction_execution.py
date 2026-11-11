import numpy as np
import pytest
from sciona.causal_prediction_execution import build_one_step_prediction_branch
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer.runner import CDGExecutionSession


@pytest.mark.asyncio
async def test_probability_branch_executes_with_permuted_classes(tmp_path,monkeypatch):
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    graph=build_one_step_prediction_branch()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],'cdg_edges':[{**e,'version_id':'test'} for e in edges]},version_id='test',content_hash=digest,require_execution_envelope=True)
    session=CDGExecutionSession(None,'synthetic-causal','branch')
    # Columns are +1,-1,0: unpaired scores .6,-.2 become .4,-.4.
    result=await session.execute({'probabilities':[[.7,.1,.2],[.2,.4,.4]],'class_labels':[1,-1,0]},cdg=restored)
    assert result['status']=='completed'
    np.testing.assert_allclose(np.load(tmp_path/'branch'/'paired_score'/'out_causal_scores.npy'),[.4,-.4])


@pytest.mark.asyncio
async def test_three_systems_use_distinct_ports_and_match_scalar_reference(tmp_path,monkeypatch):
    from sciona.causal_prediction_execution import build_three_system_prediction_graph
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    graph=build_three_system_prediction_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],'cdg_edges':[{**e,'version_id':'test'} for e in edges]},version_id='test',content_hash=digest,require_execution_envelope=True)
    rng=np.random.default_rng(438)
    p=rng.dirichlet(np.ones(3),512)
    ind,left,right=rng.uniform(0,1,(3,512))
    direction=rng.uniform(-1,1,512)
    weights=[.383,.370,.247]
    expected=[]
    for i in range(0,512,2):
        id_score=(ind[i]+ind[i+1])/2*(direction[i]-direction[i+1])/2
        lr_score=((left[i]-right[i])-(left[i+1]-right[i+1]))/2
        one=((p[i,2]-p[i,0])-(p[i+1,2]-p[i+1,0]))/2
        score=weights[0]*id_score+weights[1]*lr_score+weights[2]*one
        expected.extend([score,-score])
    result=await CDGExecutionSession(None,'synthetic-three-system','three').execute({'probabilities':p,'class_labels':np.array([-1,0,1]),'independence_scores':ind,'direction_scores':direction,'left_scores':left,'right_scores':right,'weights':np.array(weights)},cdg=graph)
    assert result['status']=='completed'
    np.testing.assert_allclose(np.load(tmp_path/'three'/'ensemble'/'out_causal_scores.npy'),expected,atol=1e-15,rtol=1e-12)


@pytest.mark.asyncio
async def test_trained_classifier_executes_before_score_conversion(tmp_path,monkeypatch):
    from sklearn.ensemble import GradientBoostingClassifier
    from sciona.causal_prediction_execution import build_trained_one_step_prediction_graph
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    rng=np.random.default_rng(702)
    train=rng.normal(size=(90,4))
    labels=np.tile([-1,0,1],30)
    model=GradientBoostingClassifier(n_estimators=8,max_depth=2,random_state=9).fit(train,labels)
    inputs=rng.normal(size=(20,4))
    p=model.predict_proba(inputs)
    raw=p[:,list(model.classes_).index(1)]-p[:,list(model.classes_).index(-1)]
    expected=np.empty(20)
    for i in range(0,20,2):expected[i],expected[i+1]=(raw[i]-raw[i+1])/2,(raw[i+1]-raw[i])/2
    graph=build_trained_one_step_prediction_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],'cdg_edges':[{**e,'version_id':'test'} for e in edges]},version_id='test',content_hash=digest,require_execution_envelope=True)
    result=await CDGExecutionSession(None,'synthetic-trained','trained').execute({'estimator':model,'X':inputs},cdg=graph)
    assert result['status']=='completed'
    np.testing.assert_allclose(np.load(tmp_path/'trained'/'paired_score'/'out_causal_scores.npy'),expected,atol=1e-15)


@pytest.mark.asyncio
async def test_five_trained_models_feed_three_system_ensemble(tmp_path,monkeypatch):
    from sklearn.ensemble import GradientBoostingClassifier
    from sciona.causal_prediction_execution import build_trained_three_system_prediction_graph
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    rng=np.random.default_rng(990)
    train=rng.normal(size=(90,4));y=np.tile([-1,0,1],30)
    targets=[y,(y!=0).astype(int),y[y!=0],(y==1).astype(int),(y==-1).astype(int)]
    models=[GradientBoostingClassifier(n_estimators=5,max_depth=2,random_state=i).fit(train[y!=0] if i==2 else train,t) for i,t in enumerate(targets)]
    X=rng.normal(size=(64,4))
    ps=[model.predict_proba(X) for model in models]
    binary=[p[:,list(m.classes_).index(1)] for p,m in zip(ps[1:],models[1:])]
    raw=ps[0][:,2]-ps[0][:,0]
    expected=[];weights=np.array([.383,.370,.247])
    for i in range(0,len(X),2):
        ind,direction,left,right=binary
        a=(ind[i]+ind[i+1])*(direction[i]-direction[i+1])/4
        b=((left[i]-right[i])-(left[i+1]-right[i+1]))/2
        c=(raw[i]-raw[i+1])/2
        value=.383*a+.370*b+.247*c
        expected.extend([value,-value])
    graph=build_trained_three_system_prediction_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],'cdg_edges':[{**e,'version_id':'test'} for e in edges]},version_id='test',content_hash=digest,require_execution_envelope=True)
    inputs=dict(zip(['one_step_model','independence_model','direction_model','left_model','right_model'],models))
    inputs.update(X=X,weights=weights)
    result=await CDGExecutionSession(None,'synthetic-five-model','five').execute(inputs,cdg=graph)
    assert result['status']=='completed'
    np.testing.assert_allclose(np.load(tmp_path/'five'/'ensemble'/'out_causal_scores.npy'),expected,atol=1e-15)

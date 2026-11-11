import numpy as np
import pytest
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_training import causal_training_partitions,train_causal_classifiers


def test_training_roles_filter_and_weight_independently():
    X=np.arange(24,dtype=float).reshape(12,2)
    y=np.array([1,-1,0,0,1,-1,0,0,1,-1,1,-1])
    parts=causal_training_partitions(X,y)
    np.testing.assert_array_equal(parts[2][0],X[[0,1,4,5,8,9,10,11]])
    np.testing.assert_array_equal(parts[2][1],[1,-1,1,-1,1,-1,1,-1])
    np.testing.assert_array_equal(parts[2][2],np.ones(8))
    assert parts[0][2] is None
    for index,target in [(1,(y!=0).astype(int)),(3,(y==1).astype(int)),(4,(y==-1).astype(int))]:
        np.testing.assert_array_equal(parts[index][1],target)
        weight=parts[index][2]
        assert weight[target==0].sum()==pytest.approx(weight[target==1].sum())
    np.testing.assert_array_equal(y,[1,-1,0,0,1,-1,0,0,1,-1,1,-1])


@pytest.mark.parametrize('labels',[[1,1,0,0,-1,-1],[1,-1,1,-1,1,-1],[1.,-1.,0.,0.,1.,-1.]])
def test_invalid_training_labels_rejected(labels):
    with pytest.raises(ValueError):train_causal_classifiers(np.ones((6,2)),np.array(labels),n_estimators=2)


@pytest.mark.asyncio
async def test_training_to_ensemble_executes_without_supplied_models(tmp_path,monkeypatch):
    from sciona.causal_prediction_execution import build_causal_training_prediction_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
    from sciona.visualizer.runner import CDGExecutionSession
    from sklearn.ensemble import GradientBoostingClassifier
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    rng=np.random.default_rng(1207)
    train=rng.normal(size=(36,4));y=np.tile([1,-1,0,0,-1,1],6)
    X=rng.normal(size=(16,4))
    # Independent public-API training with explicit role targets and weights.
    models=[]
    targets=[y,(y!=0).astype(int),y[y!=0],(y==1).astype(int),(y==-1).astype(int)]
    for index,target in enumerate(targets):
        weights=None if index==0 else np.ones(len(target))
        if index in [1,3,4]:weights[target==0]=(target==1).sum()/(target==0).sum()
        model=GradientBoostingClassifier(n_estimators=3,max_depth=2,min_samples_split=8,learning_rate=.1,random_state=1)
        models.append(model.fit(train[y!=0] if index==2 else train,target,sample_weight=weights))
    p=[m.predict_proba(X) for m in models]
    values=[a[:,1] for a in p[1:]]
    expected=[]
    for i in range(0,len(X),2):
        ind,direction,left,right=values
        one=((p[0][i,2]-p[0][i,0])-(p[0][i+1,2]-p[0][i+1,0]))/2
        score=.383*(ind[i]+ind[i+1])*(direction[i]-direction[i+1])/4+.370*((left[i]-right[i])-(left[i+1]-right[i+1]))/2+.247*one
        expected.extend([score,-score])
    graph=build_causal_training_prediction_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],'cdg_edges':[{**e,'version_id':'test'} for e in edges]},version_id='test',content_hash=digest,require_execution_envelope=True)
    result=await CDGExecutionSession(None,'synthetic-training','training').execute({'training_features':train,'training_labels':y,'X':X,'weights':np.array([.383,.370,.247]),'n_estimators':3,'max_depth':2},cdg=graph)
    assert result['status']=='completed'
    np.testing.assert_allclose(np.load(tmp_path/'training'/'ensemble'/'out_causal_scores.npy'),expected,atol=1e-15)

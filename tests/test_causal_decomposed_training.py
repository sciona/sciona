import numpy as np
import pytest
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_partitions import prepare_causal_training_partitions, witness_prepare_causal_training_partitions
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_training import train_causal_classifiers
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_predictions import causal_classifier_predictions
from sciona.ghost.abstract import AbstractArray


def test_partition_ports_preserve_roles_and_have_independent_storage():
    X=np.arange(24.).reshape(12,2)
    y=np.array([1,-1,0,0,1,-1,0,0,1,-1,1,-1])
    parts=prepare_causal_training_partitions(X,y)
    assert len(parts)==15 and parts[2] is None
    np.testing.assert_array_equal(parts[6],X[y!=0])
    np.testing.assert_array_equal(parts[7],y[y!=0])
    np.testing.assert_array_equal(parts[8],np.ones(8))
    for start,expected in [(3,y!=0),(9,y==1),(12,y==-1)]:
        np.testing.assert_array_equal(parts[start+1],expected.astype(int))
        weights=parts[start+2]
        assert weights[~expected].sum()==pytest.approx(weights[expected].sum())
    parts[0][:]=999
    np.testing.assert_array_equal(parts[3],X)
    assert X[0,0]==0
    metadata=witness_prepare_causal_training_partitions(X,y)
    assert metadata[6].shape==(8,2) and metadata[2] is None


def test_partition_witness_rejects_unknown_direction_cardinality():
    with pytest.raises(ValueError,match='Concrete'):
        witness_prepare_causal_training_partitions(AbstractArray(shape=(12,2),dtype='float64'),AbstractArray(shape=(12,),dtype='int64'))


@pytest.mark.parametrize('labels', [[1,1,0,0,-1,-1],[1,-1,1,-1,1,-1],[1.,-1.,0.,0.,1.,-1.]])
def test_partition_adapter_preserves_label_contract_rejections(labels):
    with pytest.raises(ValueError):
        prepare_causal_training_partitions(np.ones((6,2)),np.asarray(labels))


@pytest.mark.asyncio
async def test_full_default_five_fit_graph_matches_previous_ensemble(tmp_path,monkeypatch):
    from sciona.causal_decomposed_training import build_explicit_causal_training_graph
    from sciona.atoms.causal_inference.feature_primitives.selected_features import prepare_causal_feature_matrices
    from sciona.services.execution_graph_codec import encode_execution_graph
    from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
    from sciona.visualizer.runner import CDGExecutionSession
    from scripts.validate_causal_execution import synthetic_pairs
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    rng=np.random.default_rng(733)
    train,train_types=synthetic_pairs(rng,18)
    query,query_types=synthetic_pairs(rng,9)
    labels=np.tile([-1,0,1],6)
    features,targets,X=prepare_causal_feature_matrices(train,query,train_types,query_types,labels)
    models=train_causal_classifiers(features,targets)
    probabilities,classes,ind,direction,left,right=causal_classifier_predictions(X,*models)
    raw=probabilities[:,list(classes).index(1)]-probabilities[:,list(classes).index(-1)]
    expected=[]
    for i in range(0,len(X),2):
        value=(.383*(ind[i]+ind[i+1])*(direction[i]-direction[i+1])/4
               +.370*((left[i]-right[i])-(left[i+1]-right[i+1]))/2
               +.247*(raw[i]-raw[i+1])/2)
        expected.extend([value,-value])
    graph=build_explicit_causal_training_graph()
    assert len(graph.nodes)==13
    assert len([n for n in graph.nodes if n.node_id.startswith('fit_')])==5
    digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],
        'cdg_edges':[{**e,'version_id':'test'} for e in edges]},version_id='test',content_hash=digest,require_execution_envelope=True)
    result=await CDGExecutionSession(None,'synthetic-explicit-causal','explicit').execute({
        'training_pairs':train,'prediction_pairs':query,'training_types':train_types,
        'prediction_types':query_types,'pair_labels':labels,'weights':np.array([.383,.370,.247])},cdg=restored)
    assert result['status']=='completed'
    assert len(result['trace'])==13
    assert {entry['node_id'] for entry in result['trace']}=={node.node_id for node in graph.nodes}
    assert all(not entry['cached'] for entry in result['trace'])
    np.testing.assert_allclose(np.load(tmp_path/'explicit'/'ensemble'/'out_causal_scores.npy'),expected,rtol=1e-12,atol=1e-15)

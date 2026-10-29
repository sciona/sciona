"""Synthetic independent-API and source-role checks for the reusable fit."""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.weighted_fit import fit_weighted_gradient_boosting
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_training import causal_training_partitions, train_causal_classifiers


@pytest.mark.parametrize('weighted', [False, True])
def test_arbitrary_multiclass_labels_match_public_api_without_mutation(weighted):
    rng = np.random.default_rng(117)
    X = rng.normal(size=(48,6))
    labels = np.tile([12,37,91],16)
    weights = rng.uniform(.2,2,size=48) if weighted else None
    originals = [a.copy() if a is not None else None for a in [X,labels,weights]]
    options = dict(n_estimators=7,max_depth=2,min_samples_split=4,learning_rate=.07,random_state=41)
    expected = GradientBoostingClassifier(loss='log_loss',**options).fit(X,labels,sample_weight=weights)
    actual = fit_weighted_gradient_boosting(X,labels,weights,**options)
    query = rng.normal(size=(13,6))
    np.testing.assert_array_equal(actual.classes_,[12,37,91])
    np.testing.assert_array_equal(actual.predict_proba(query),expected.predict_proba(query))
    for before,after in zip(originals,[X,labels,weights]):
        if before is not None: np.testing.assert_array_equal(before,after)


def test_all_five_source_roles_match_at_full_default_500_estimators():
    rng = np.random.default_rng(912)
    X = rng.normal(size=(36,5))
    labels = np.tile([1,-1,0,0,-1,1],6)
    query = rng.normal(size=(14,5))
    reference = train_causal_classifiers(X,labels)
    for expected,(features,target,weight) in zip(reference,causal_training_partitions(X,labels)):
        actual = fit_weighted_gradient_boosting(features,target,weight)
        assert actual.n_estimators_ == 500
        np.testing.assert_array_equal(actual.classes_,expected.classes_)
        np.testing.assert_array_equal(actual.predict_proba(query),expected.predict_proba(query))


@pytest.mark.parametrize('change', [
    {'X': [[np.nan],[1.],[2.],[3.]]}, {'X': np.ones((4,0))},
    {'labels': [1,1,1,1]}, {'labels': [1.,2.,1.,2.]}, {'labels': [1,2]},
    {'sample_weight': [0,1,1,1]}, {'sample_weight': [-1,1,1,1]},
    {'sample_weight': [np.inf,1,1,1]}, {'sample_weight': [1,1]},
    {'n_estimators': True}, {'n_estimators': 0}, {'max_depth': 0},
    {'min_samples_split': 1}, {'random_state': -1}, {'random_state': 2**32},
    {'learning_rate': np.nan}, {'learning_rate': '0.1'}, {'learning_rate': True},
])
def test_invalid_contract_rejected(change):
    inputs = dict(X=np.arange(8.).reshape(4,2),labels=np.array([1,2,1,2]),n_estimators=2)
    inputs.update(change)
    with pytest.raises(ValueError): fit_weighted_gradient_boosting(**inputs)


@pytest.mark.asyncio
async def test_fit_model_flows_through_stored_graph_into_prediction(tmp_path, monkeypatch):
    from sciona.architect.handoff import CDGExport
    from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
    from sciona.services.execution_graph_codec import encode_execution_graph
    from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
    from sciona.visualizer.runner import CDGExecutionSession
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    prefix = 'sciona.atoms.ml.sklearn.ensemble.gradient_boosting.'
    array = 'numpy.ndarray'
    graph = CDGExport(nodes=[
        AlgorithmicNode(node_id='fit',name='Fit weighted classifier',description='Generic single-model fit',
            concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=prefix+'weighted_fit.fit_weighted_gradient_boosting',
            inputs=[IOSpec(name=n,type_desc=t) for n,t in [('X',array),('labels',array),('sample_weight',array),('n_estimators','int')]],
            outputs=[IOSpec(name='model',type_desc='object')]),
        AlgorithmicNode(node_id='predict',name='Predict labelled probabilities',description='Generic model prediction',
            concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=prefix+'prediction.gradient_boosting_class_probabilities',
            inputs=[IOSpec(name='estimator',type_desc='object'),IOSpec(name='X',type_desc=array)],
            outputs=[IOSpec(name='probabilities',type_desc=array),IOSpec(name='class_labels',type_desc=array)]),
    ],edges=[DependencyEdge(source_id='fit',target_id='predict',output_name='model',input_name='estimator',source_type='object',target_type='object')])
    digest,nodes,edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],
        'cdg_edges':[{**e,'version_id':'test'} for e in edges]},version_id='test',content_hash=digest,require_execution_envelope=True)
    rng = np.random.default_rng(511)
    X = rng.normal(size=(30,4)); labels = np.tile([4,8,19],10)
    weights = rng.uniform(.5,1.5,size=30)
    # Shared X deliberately checks the runner's model handoff; separate train/query
    # matrices are covered by the direct API parity tests above.
    expected = GradientBoostingClassifier(loss='log_loss',n_estimators=6,max_depth=9,
        min_samples_split=8,learning_rate=.1,random_state=1).fit(X,labels,sample_weight=weights)
    result = await CDGExecutionSession(None,'synthetic-generic-weighted-fit','fit-flow').execute(
        {'X':X,'labels':labels,'sample_weight':weights,'n_estimators':6},cdg=restored)
    assert result['status'] == 'completed'
    np.testing.assert_array_equal(np.load(tmp_path/'fit-flow'/'predict'/'out_probabilities.npy'),expected.predict_proba(X))
    np.testing.assert_array_equal(np.load(tmp_path/'fit-flow'/'predict'/'out_class_labels.npy'),expected.classes_)

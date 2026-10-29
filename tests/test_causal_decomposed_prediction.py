import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_model_roles import validate_causal_model_roles,witness_validate_causal_model_roles


def models():
    X=np.arange(36.).reshape(12,3)
    targets=[np.tile([-1,0,1],4),np.tile([0,1],6),np.tile([-1,1],6),np.tile([0,1],6),np.tile([0,1],6)]
    return [GradientBoostingClassifier(n_estimators=2,max_depth=1,random_state=1).fit(X,y) for y in targets]


def test_role_guard_returns_models_without_predicting(monkeypatch):
    values=models()
    def forbidden(*args,**kwargs):raise AssertionError('role guard must not predict')
    monkeypatch.setattr(GradientBoostingClassifier,'predict_proba',forbidden)
    result=validate_causal_model_roles(*values)
    assert all(actual is original for actual,original in zip(result,values))


@pytest.mark.parametrize('role',[0,1,2,3,4])
def test_incorrect_role_label_set_rejected(role):
    values=models()
    values[role]=values[1] if role in [0,2] else values[2]
    with pytest.raises(ValueError,match='role'):validate_causal_model_roles(*values)


def test_abstract_role_witness_rejects_unproven_classes():
    with pytest.raises(ValueError,match='Concrete'):witness_validate_causal_model_roles(*[{}]*5)


@pytest.mark.asyncio
async def test_five_predictions_preserve_full_default_ensemble(tmp_path,monkeypatch):
    # Reuse the unchanged numerical oracle and input population, replacing only
    # graph acquisition and the expected executed-node count.
    import runpy
    from pathlib import Path
    import sciona.causal_decomposed_training as training
    from sciona.causal_decomposed_prediction import build_explicit_causal_prediction_graph
    graph=build_explicit_causal_prediction_graph()
    assert len(graph.nodes)==22
    assert len([n for n in graph.nodes if n.node_id.startswith('predict_')])==5
    assert len([n for n in graph.nodes if n.node_id.startswith('positive_')])==4
    monkeypatch.setattr(training,'build_explicit_causal_training_graph',lambda:graph)
    oracle=runpy.run_path(str(Path(__file__).with_name('test_causal_decomposed_training.py')))
    await oracle['check_full_default_five_fit_graph'](tmp_path,monkeypatch,expected_nodes=22)

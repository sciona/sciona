"""Control-flow tests only; these do not claim model training or inference."""
import json
from pathlib import Path

import numpy as np
import pytest
import sciona.tgs_workflow as workflow


def test_all66_actions_are_persisted_in_order(monkeypatch):
    plan = json.loads((Path(__file__).resolve().parents[1] / 'docs/reviews/competition_tgs_training_plan.json').read_text())
    executed, recorded = [], []
    def fit(key, *args, **kwargs):
        executed.append(key)
        return dict(fit=key)
    def predict(stage, *args, **kwargs):
        executed.append(f'round{stage}')
        return dict(scores=np.zeros((1,101,101)))
    monkeypatch.setattr(workflow, 'run_fit', fit)
    monkeypatch.setattr(workflow, 'predict_round', predict)
    monkeypatch.setattr(workflow, 'complete_round', lambda stage, *a, **k: dict(stage=stage))
    monkeypatch.setattr(workflow, 'TGSResNeXt50', lambda: object())
    monkeypatch.setattr(workflow, 'TGSResNet34', lambda variant: object())
    population = dict(query_images=np.zeros((1,101,101)),query_nonconstant=np.ones(1,dtype=bool),
                      labeled_images=np.zeros((1,101,101)),labeled_masks=np.zeros((1,101,101)))
    result = workflow.run_workflow(plan,dict(keras=population,pytorch=population),{},None,seed=71,
                                  record=lambda state: recorded.append((len(state['fits']),len(state['rounds']))))
    assert len(executed) == len(recorded) == 66
    assert recorded[-1] == (63,3)
    assert len(result['fits']) == 63
    assert all(sum(b) == sum(a)+1 for a,b in zip(recorded,recorded[1:]))


def test_failed_persistence_stops_before_next_fit(monkeypatch):
    plan = json.loads((Path(__file__).resolve().parents[1] / 'docs/reviews/competition_tgs_training_plan.json').read_text())
    calls=[]
    monkeypatch.setattr(workflow,'run_fit',lambda key,*a,**k: calls.append(key) or {})
    def fail(state):
        raise OSError('synthetic persistence failure')
    with pytest.raises(OSError,match='persistence failure'):
        workflow.run_workflow(plan,dict(keras={}),{},None,seed=1,record=fail)
    assert len(calls)==1

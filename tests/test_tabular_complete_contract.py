"""Synthetic strict boundary and execution checks for expanded tabular features."""
import json
import pytest
from scripts.tabular_ensemble_synthetic import payload as legacy_payload
from tests.test_tabular_complete_training import POLICY
from sciona import tabular_complete_contract as contract


def payload():
    p = legacy_payload(); p['version'] = 2
    p['controls'] = dict(seed=42, trees=8, max_depth=6, min_leaf=1,
        features=dict(POLICY, group_aggregates=[dict(key_column=0, value_columns=[0],
        statistics=['mean'], unknown_value=0.)], text_topics=[dict(column=0, topics=2,
        max_features=20, iterations=4, lowercase=False, seed=42)], entity_embeddings=[
        dict(columns=[0], dimension=2, iterations=1000, regularization=.1, tolerance=1e-5, seed=42)],
        selection=dict(method='rfe', keep=3, step=3)))
    return p


def test_immutable_prepared_boundary_runs_complete_recipe():
    p = payload(); prepared = contract.prepare(p)
    p['training']['numeric'][0][0] = -999
    assert json.loads(prepared.configuration)['training']['numeric'][0][0] == 0
    result = contract.execute(prepared)
    json.dumps(result, allow_nan=False)
    assert (result['training_rows'], result['calibration_rows'], result['query_rows']) == (12, 4, 2)
    assert (result['folds'], result['oof_rows'], result['models'], result['forest_trees']) == (3, 12, 2, 8)
    assert result['classes'] == [int(p >= .5) for p in result['probabilities']]


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(version=1), lambda p: p.update(extra=True),
    lambda p: p['controls']['features'].update(selection={'method': 'rfe', 'keep': True, 'step': 1}),
    lambda p: p['controls']['features'].update(log_columns=[99]),
    lambda p: p['controls']['features'].update(smoothing=-1),
    lambda p: p['controls']['features']['text_topics'][0].update(lowercase=1),
    lambda p: p['controls']['features']['entity_embeddings'][0].update(tolerance=0),
    lambda p: p['controls']['features']['group_aggregates'][0].update(statistics=['unknown']),
    lambda p: p['controls']['features'].update(max_mode_fraction=float('nan')),
    lambda p: p['query']['groups'].__setitem__(0, p['training']['groups'][0]),
    lambda p: p['training'].update(folds=[0]*12),
])
def test_invalid_configuration_fails_before_model_fitting(mutate, monkeypatch):
    p = payload(); mutate(p)
    def forbidden(*args, **kwargs): raise AssertionError('Must reject before fitting')
    monkeypatch.setattr(contract, 'fit_complete_stacker', forbidden)
    with pytest.raises(ValueError):
        contract.execute(contract.PreparedComplete(json.dumps(p)))

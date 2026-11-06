"""Synthetic complete-lifecycle parity and stage isolation checks."""
import copy
import json
import numpy as np
import pytest
import torch

from scripts.physical_operator_synthetic import payload
from sciona import physical_operator_stages as stages
from sciona import physical_operator_contract as retained


@pytest.mark.parametrize('change', ['baseline', 'repeated_query', 'no_smoothing'])
def test_decomposed_full_lifecycle_matches_retained(change, monkeypatch):
    p = payload()
    if change == 'repeated_query':
        for field in ('states', 'diffusivity', 'elapsed', 'groups'):
            p['query'][field].append(copy.deepcopy(p['query'][field][0]))
    if change == 'no_smoothing':
        p['controls']['smoothing'] = 0
    before = copy.deepcopy(p)
    expected = retained.execute(retained.prepare(p))
    prepared = stages.prepare(p)
    fitted = stages.fit(prepared)
    weights = {k:v.clone() for k,v in fitted.fitted.model.state_dict().items()}
    history = copy.deepcopy(fitted.fitted.history)
    def forbidden(*args, **kwargs):
        raise AssertionError('Later stages must not refit')
    monkeypatch.setattr(stages, 'train', forbidden)
    predicted = stages.predict(fitted)
    projected = stages.project(predicted)
    actual = stages.finish(projected)
    assert predicted.fitted_state is fitted and projected.fitted_state is fitted
    assert p == before and fitted.fitted.history == history
    assert all(torch.equal(v, fitted.fitted.model.state_dict()[k]) for k,v in weights.items())
    assert set(actual) == set(expected)
    for field in actual:
        np.testing.assert_allclose(actual[field], expected[field], rtol=1e-12, atol=1e-12)
    json.dumps(actual, allow_nan=False)
    values = np.asarray(actual['predictions'])
    assert (values >= 0).all()
    np.testing.assert_allclose(values.sum(1), np.asarray(p['query']['states']).sum(1), atol=1e-12)


def test_prepared_configuration_is_detached_and_revalidated(monkeypatch):
    p = payload()
    prepared = stages.prepare(p)
    p['query']['groups'][0] = p['training']['groups'][0]
    assert json.loads(prepared.configuration)['query']['groups'] != p['query']['groups']
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid configuration must fail before training')
    monkeypatch.setattr(stages, 'train', forbidden)
    with pytest.raises(ValueError):
        stages.fit(stages.PreparedOperator(json.dumps(p)))


@pytest.mark.parametrize('operation', [stages.fit, stages.predict, stages.project, stages.finish])
def test_wrong_stage_rejected(operation):
    with pytest.raises(ValueError):
        operation({})

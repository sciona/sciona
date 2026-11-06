"""Synthetic full-computation parity and boundaries for explicit Santa stages."""
from dataclasses import replace
import numpy as np
import pytest

from sciona import decomposed_santa_runtime as runtime
from sciona.santa_execution import prepare, execute
from sciona.santa_agent_features import choose_action, exploration_ratio
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from scripts.santa_synthetic import payload


@pytest.fixture(scope='module')
def fitted():
    prepared = prepare(payload())
    return prepared, runtime.fit(prepared)


def test_full_reconstruction_exact(fitted):
    prepared, models = fitted
    predicted = runtime.predictions(prepared, models)
    values, coefficients = runtime.blend_inputs(prepared, predicted)
    result = runtime.result(prepared, models, ordered_weighted_sum(values, coefficients))
    assert result == execute(prepared)
    assert result['models'] == 2
    assert result['training_rows'] > 0 and result['validation_rows'] > 0
    assert set(result['model_evaluation']) == {'normal','transformed'}


@pytest.mark.parametrize('steps', [0, 1, 100, 1999])
def test_blend_and_action_exact_across_schedule(fitted, steps):
    _, models = fitted
    inputs = payload()
    inputs['query']['history'] = [dict(own_action=i % 100, opponent_action=(i+1) % 100, reward=0)
                                  for i in range(steps)]
    prepared = prepare(inputs)
    predicted = dict(normal=np.linspace(2,80,100), transformed=np.linspace(1,7,100))
    values, coefficients = runtime.blend_inputs(prepared, predicted)
    actual = runtime.result(prepared, models, ordered_weighted_sum(values, coefficients))
    history = runtime._history(inputs['query'])
    expected = choose_action(predicted['normal'], predicted['transformed'], history,
                             seed=inputs['controls']['decision_seed'])
    assert {key: actual[key] for key in expected} == expected
    assert coefficients[0] == exploration_ratio(steps)
    if steps == 0:
        assert coefficients[1] < 0  # Extrapolation must not become a convex blend.


@pytest.mark.parametrize('initial', [np.zeros(100), -np.ones(100)])
def test_nonpositive_fallback(fitted, initial):
    prepared, models = fitted
    assert runtime.result(prepared, models, initial)['action'] == 0


@pytest.mark.parametrize('fault', ['missing', 'short', 'zero', 'nan'])
def test_invalid_predictions_rejected(fitted, fault):
    prepared, _ = fitted
    predicted = dict(normal=np.ones(100), transformed=np.ones(100))
    if fault == 'missing': predicted.pop('normal')
    if fault == 'short': predicted['normal'] = np.ones(99)
    if fault == 'zero': predicted['transformed'][0] = 0
    if fault == 'nan': predicted['normal'][0] = np.nan
    with pytest.raises(ValueError): runtime.blend_inputs(prepared, predicted)


def test_modified_prepared_rejected(fitted):
    prepared, _ = fitted
    with pytest.raises(ValueError, match='changed'):
        runtime.fit(replace(prepared, fingerprint='0'*64))


@pytest.mark.parametrize('initial', [np.zeros(99), np.full(100, np.inf)])
def test_invalid_blend_rejected(fitted, initial):
    with pytest.raises(ValueError): runtime.result(*fitted, initial)


def test_missing_models_rejected(fitted):
    with pytest.raises(ValueError): runtime.predictions(fitted[0], object())


def test_query_changes_do_not_change_fitting(fitted):
    _, models = fitted
    inputs = payload()
    inputs['query']['history'] = []
    inputs['controls']['decision_seed'] += 1
    alternate = runtime.fit(prepare(inputs))
    assert models.normal.model_to_string() == alternate.normal.model_to_string()
    assert models.transformed.model_to_string() == alternate.transformed.model_to_string()
    assert models.evaluation == alternate.evaluation and models.rows == alternate.rows

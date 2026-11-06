"""Synthetic source parity and domain-independent central rank reduction."""
import numpy as np
import pytest

from sciona.atoms.numerical.statistics.central_mean import central_rank_mean
from sciona.champs_ensemble import MODEL_ORDER, COUPLING_TYPES, blend_predictions
from sciona.decomposed_champs_ensemble import ensemble_inputs, result


def candidate(predictions, kinds):
    keys, values, retained = ensemble_inputs(predictions, kinds)
    return result(keys, central_rank_mean(values, retained))


@pytest.mark.parametrize('rows,retained,columns', [(3, 1, 2), (4, 2, 7), (9, 5, 0), (13, 9, 4), (1, 1, 2)])
def test_generic_sizes_order_and_no_mutation(rows, retained, columns):
    values = np.random.default_rng(732).normal(size=(rows, columns))
    before = values.copy()
    values.flags.writeable = False
    actual = central_rank_mean(values, retained)
    start = (rows - retained) // 2
    expected = np.array([np.sort(values[:, column])[start:start + retained].mean()
                         for column in range(columns)], dtype=np.float64)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, before)
    assert not np.shares_memory(actual, values)


def test_generic_odd_discard_policy():
    # No molecular categories, model count, or fixed five-value window.
    assert central_rank_mean(np.array([[90], [3], [1], [2]]), 1).tolist() == [2.]
    assert central_rank_mean(np.array([[90], [3], [1], [2]]), 2).tolist() == [2.5]


@pytest.mark.parametrize('mode', ['ordinary', 'wide_exponents', 'subnormal', 'ties', 'empty'])
def test_exact_source_selection_alignment_and_arithmetic(mode):
    rng = np.random.default_rng(432)
    keys = [f'synthetic-{i}' for i in range(40)] if mode != 'empty' else []
    kinds = {key: COUPLING_TYPES[i % 8] for i, key in enumerate(keys)}
    values = rng.normal(size=(13, len(keys)))
    if mode == 'wide_exponents':
        values *= np.power(10., rng.integers(-280, 280, size=values.shape))
    elif mode == 'subnormal':
        values = rng.integers(-15, 15, size=values.shape) * np.nextafter(0., 1.)
    elif mode == 'ties':
        values = np.round(values)
    predictions = {name: dict(reversed(list(zip(keys, row)))) for name, row in zip(MODEL_ORDER, values)}
    actual, expected = candidate(predictions, kinds), blend_predictions(predictions, kinds)
    assert list(actual) == list(expected)
    assert actual == expected
    np.testing.assert_array_equal(np.signbit(list(actual.values())), np.signbit(list(expected.values())))


@pytest.mark.parametrize('fault', ['nan', 'infinity', 'bool', 'missing_model', 'extra_model', 'missing_record', 'unknown_type', 'overflow'])
def test_source_failures_preserved(fault):
    predictions = {name: {'synthetic': float(i)} for i, name in enumerate(MODEL_ORDER)}
    kinds = {'synthetic': COUPLING_TYPES[0]}
    if fault in ('nan', 'infinity', 'bool'):
        predictions[MODEL_ORDER[0]]['synthetic'] = {'nan': np.nan, 'infinity': np.inf, 'bool': True}[fault]
    elif fault == 'missing_model':
        predictions.pop(MODEL_ORDER[0])
    elif fault == 'extra_model':
        predictions['extra'] = {'synthetic': 0.}
    elif fault == 'missing_record':
        predictions[MODEL_ORDER[0]] = {}
    elif fault == 'unknown_type':
        kinds['synthetic'] = 'unsupported'
    else:
        predictions = {name: {'synthetic': 1e308} for name in MODEL_ORDER}
    with pytest.raises(ValueError):
        blend_predictions(predictions, kinds)
    with pytest.raises(ValueError):
        candidate(predictions, kinds)


@pytest.mark.parametrize('values,retained', [(np.zeros((0, 3)), 1), (np.ones(3), 1),
    (np.ones((2, 1), dtype=bool), 1), (np.ones((2, 1)), True), (np.ones((2, 1)), 0),
    (np.ones((2, 1)), 3), (np.array([[np.nan]]), 1), (np.array([[1j]]), 1)])
def test_generic_invalid_contracts(values, retained):
    with pytest.raises(ValueError):
        central_rank_mean(values, retained)

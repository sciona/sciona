"""Synthetic group-statistic contracts; no source dataset fixtures."""
import numpy as np
import pytest

from sciona.tabular_group_features import fit_group_features


def test_hand_calculated_statistics_and_row_alignment():
    state, training = fit_group_features([[2., 10.], [8., 20.], [4., 30.]], ['b', 'a', 'b'],
        statistics=['count', 'sum', 'mean', 'var', 'std', 'min', 'max'], unknown_value=-9.)
    expected_b = [2, 6, 3, 1, 1, 2, 4, 2, 40, 20, 100, 10, 10, 30]
    expected_a = [1, 8, 8, 0, 0, 8, 8, 1, 20, 20, 0, 0, 20, 20]
    np.testing.assert_array_equal(training, [expected_b, expected_a, expected_b])
    np.testing.assert_array_equal(state.transform(['new', 'b', 'b', 'a']),
        [[-9]*14, expected_b, expected_b, expected_a])


def test_query_batch_does_not_mutate_fitted_maps_or_depend_on_frequency():
    values, keys = [[2.], [4.]], ['x', 'x']
    state, _ = fit_group_features(values, keys, statistics=['mean'], unknown_value=0.)
    values[0][0] = 1000
    keys[0] = 'changed'
    queried = state.transform(['x']*20 + ['unknown'])
    queried[0, 0] = -100
    np.testing.assert_array_equal(state.transform(['x', 'unknown']), [[3.], [0.]])
    with pytest.raises(TypeError):
        state.mapping['x'] = (100.,)


def test_aggregate_keys_preserve_exact_strings():
    state, _ = fit_group_features([[1.], [2.], [3.]], ['', 'a|b', 'a'],
        statistics=['sum'], unknown_value=0.)
    np.testing.assert_array_equal(state.transform(['a', '', 'a|b']), [[3.], [1.], [2.]])


@pytest.mark.parametrize('values,keys,statistics,fill', [
    ([], [], ['mean'], 0), ([[1]], ['a', 'b'], ['mean'], 0),
    ([[1], [2, 3]], ['a', 'b'], ['mean'], 0),
    ([[True]], ['a'], ['mean'], 0), ([[float('nan')]], ['a'], ['mean'], 0),
    ([[1]], [1], ['mean'], 0), ([[1]], ['a'], [], 0),
    ([[1]], ['a'], ['mean', 'mean'], 0), ([[1]], ['a'], ['median'], 0),
    ([[1]], ['a'], ['mean'], float('inf')), ([[1]], ['a'], ['mean'], True),
])
def test_rejects_ambiguous_or_nonfinite_inputs(values, keys, statistics, fill):
    with pytest.raises(ValueError):
        fit_group_features(values, keys, statistics=statistics, unknown_value=fill)

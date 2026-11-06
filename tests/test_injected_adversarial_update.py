"""Injected operations must control every normalization in the source recurrence."""
import numpy as np
import pytest

from sciona.atoms.numerical.statistics.row_scale import normalize_rows_by_scale
from sciona.adversarial_updates import update as reference
from sciona.decomposed_adversarial_injected import update


@pytest.mark.parametrize('mode,epsilon', [('non_targeted', 16.), ('targeted', 4.), ('targeted', 8.)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('momentum', [0., 1.])
def test_injected_operation_preserves_source_recurrence(mode, epsilon, dtype, momentum):
    rng = np.random.default_rng(52)
    x = rng.uniform(-.8, .8, (2, 3, 4, 3)).astype(dtype)
    g = rng.normal(size=x.shape).astype(dtype)
    prior = rng.normal(size=x.shape).astype(dtype)
    args = (x, g, prior, np.maximum(x-2*epsilon/255, -1), np.minimum(x+2*epsilon/255, 1))
    calls = []
    def selected(values, statistic):
        calls.append((values.copy(), statistic))
        return normalize_rows_by_scale(values, statistic)
    kwargs = dict(mode=mode, epsilon=epsilon, momentum=momentum)
    actual = update(*args, normalizer=selected, **kwargs)
    expected = reference(*args, **kwargs)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert len(calls) == (1 if mode == 'non_targeted' else 2)
    assert [c[1] for c in calls] == (['mean_absolute'] if mode == 'non_targeted' else ['population_std']*2)
    np.testing.assert_array_equal(calls[0][0], g.reshape(2, -1))
    if mode == 'targeted':
        expected_accumulated = momentum*prior+normalize_rows_by_scale(g.reshape(2, -1), 'population_std').reshape(x.shape)
        np.testing.assert_array_equal(calls[1][0], expected_accumulated.reshape(2, -1))


def test_injected_failure_is_not_replaced_by_a_hidden_normalizer():
    x = np.zeros((1, 1, 2, 3), dtype=np.float32)
    def selected(values, statistic):
        raise RuntimeError('selected operation failed')
    with pytest.raises(RuntimeError, match='selected operation failed'):
        update(x, np.ones_like(x), x, x-1, x+1, mode='non_targeted', epsilon=16., normalizer=selected)

"""Synthetic adapter parity and explicit unsupported-boundary rejection."""
import numpy as np
import pytest

from sciona.adversarial_updates import update as reference
from sciona.decomposed_adversarial_updates import update


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('mode,epsilon', [('non_targeted', 16.), ('targeted', 4.), ('targeted', 8.)])
@pytest.mark.parametrize('momentum', [0., 1.])
def test_exact_update_and_input_preservation(dtype, mode, epsilon, momentum):
    rng = np.random.default_rng(42)
    x = rng.uniform(-.8, .8, (2, 3, 4, 3)).astype(dtype)
    args = (x, rng.normal(size=x.shape).astype(dtype),
            rng.normal(size=x.shape).astype(dtype),
            np.maximum(x - 2 * epsilon / 255, -1),
            np.minimum(x + 2 * epsilon / 255, 1))
    before = [a.copy() for a in args]
    kwargs = dict(mode=mode, epsilon=epsilon, momentum=momentum)
    actual = update(*args, **kwargs)
    expected = reference(*args, **kwargs)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
        assert all(not np.shares_memory(a, old) for old in args)
    for a, b in zip(args, before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('fault', ['zero_gradient', 'nonfinite', 'layout', 'dtype', 'bounds', 'momentum'])
def test_rejects_undefined_or_unqualified_input(fault):
    x = np.zeros((2, 3, 4, 3), dtype=np.float32)
    g = np.arange(x.size, dtype=np.float32).reshape(x.shape)
    args = [x, g, x.copy(), np.full_like(x, -1), np.full_like(x, 1)]
    kwargs = dict(mode='targeted', epsilon=4.)
    if fault == 'zero_gradient':
        args[1] = np.zeros_like(x)
    elif fault == 'nonfinite':
        args[1][0, 0, 0, 0] = np.nan
    elif fault == 'layout':
        args[1] = np.asfortranarray(g)
    elif fault == 'dtype':
        args[1] = g.astype(np.float64)
    elif fault == 'bounds':
        args[3].fill(.1)
    elif fault == 'momentum':
        kwargs['momentum'] = -1.
    with pytest.raises(ValueError):
        update(*args, **kwargs)

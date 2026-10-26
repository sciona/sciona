"""Synthetic categorical decisions, independent Decimal oracle, source parity."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from sciona.probability_ensemble import (
    mean_probability_bag, geometric_arithmetic_probability_blend as combine,
)
from sciona.ensemble_otto_adapter import blend
from sciona.otto_final_blend import blend as historical_blend


def oracle(bags, weight, additive_weight, exponent):
    with localcontext() as ctx:
        ctx.prec = 80
        means = []
        for bag in bags:
            means.append([[sum(Decimal.from_float(float(run[row][col])) for run in bag)
                           / len(bag) for col in range(bag.shape[2])]
                          for row in range(bag.shape[1])])
        w, v, a = map(Decimal.from_float, (weight, additive_weight, exponent))
        rows = []
        for left, right, additive in zip(*means):
            raw = [w * x**a * y**(1-a) + v*z for x, y, z in zip(left, right, additive)]
            rows.append([float(x/sum(raw)) for x in raw])
        return rows


@pytest.mark.parametrize('domain,classes,counts', [
    ('synthetic_machine_fault_categories', 3, (2, 3, 4)),
    ('synthetic_land_cover_categories', 5, (4, 2, 3)),
])
def test_cross_domain_class_counts_and_independent_oracle(domain, classes, counts):
    # Constructed examples only; row/class meanings belong to the caller.
    rng = np.random.default_rng(classes)
    bags = [rng.uniform(.1, 1, (count, 7, classes)) for count in counts]
    bags = [x/x.sum(axis=2, keepdims=True) for x in bags]
    before = [x.copy() for x in bags]
    result = combine(*(mean_probability_bag(x) for x in bags), .7, .3, .4)
    np.testing.assert_allclose(result['probabilities'], oracle(bags, .7, .3, .4), rtol=2e-14, atol=1e-15)
    for x, y in zip(bags, before):
        np.testing.assert_array_equal(x, y)
    permutation = np.arange(classes)[::-1]
    permuted = combine(*(mean_probability_bag(x[:, :, permutation]) for x in bags), .7, .3, .4)
    np.testing.assert_allclose(permuted['probabilities'], np.asarray(result['probabilities'])[:, permutation])
    single = combine(*(mean_probability_bag(x[:, :1]) for x in bags), .7, .3, .4)
    np.testing.assert_array_equal(single['probabilities'][0], result['probabilities'][0])


def test_source_adapter_matches_preserved_formula():
    rng = np.random.default_rng(97)
    bags = [rng.uniform(.01, 1, (runs, 4, 9)) for runs in (250, 600, 250)]
    bags = [x/x.sum(axis=2, keepdims=True) for x in bags]
    assert blend(*bags) == historical_blend(*bags)
    with pytest.raises(ValueError):
        blend(bags[0][:-1], *bags[1:])
    with pytest.raises(ValueError):
        blend(*(np.full((n, 1, 3), 1/3) for n in (250, 600, 250)))


def test_support_endpoints_and_tie_breaking():
    left, right, additive = [[1., 0]], [[0., 1]], [[.5, .5]]
    assert combine(left, right, additive, 1., 0., 0.)['probabilities'] == right
    assert combine(left, right, additive, 1., 0., 1.)['probabilities'] == left
    assert combine(left, right, additive, 0., 1., .5)['classes'] == [0]
    with pytest.raises(ValueError, match='support'):
        combine(left, right, additive, 1., 0., .5)


@pytest.mark.parametrize('bad', [[], [[[1.]]], [[[.2, .2]]], [[[np.nan, 0.]]],
    [[[1.1, -.1]]], [[[True, False]]], [[[1+0j, 0j]]], [[['1', '0']]]])
def test_invalid_bags(bad):
    with pytest.raises(ValueError):
        mean_probability_bag(bad)


@pytest.mark.parametrize('weights', [(True, 0., .5), (1., 0., np.nan),
    (.8, .3, .5), (-.1, 1.1, .5), (.5, .5, 2.)])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        combine([[.5, .5]], [[.5, .5]], [[.5, .5]], *weights)


def test_misaligned_shapes():
    with pytest.raises(ValueError, match='shapes'):
        combine([[.5, .5]], [[.5, .5], [.5, .5]], [[.5, .5]], .7, .3, .5)

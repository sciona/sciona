"""Source-order equality with generic sums, including subnormal counterexamples."""
import numpy as np
import pytest

from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona.cassava_family_alignment import assemble
from sciona.decomposed_cassava_assembly import pair_inputs, final_inputs, result


def candidate(keys, families):
    context, values, weights = pair_inputs(keys, **families)
    paired = ordered_weighted_sum(values, weights)
    values, weights = final_inputs(context, paired)
    return result(context, ordered_weighted_sum(values, weights))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('rows', [1, 3, 19])
def test_exact_identity_alignment_and_source_order(dtype, rows):
    rng = np.random.default_rng(83)
    keys = ['synthetic-'+str(i) for i in range(rows)]
    families = {}
    for name in ('vit', 'resnext', 'efficientnet', 'cropnet'):
        values = rng.dirichlet(np.ones(5), size=rows).astype(dtype)
        order = rng.permutation(rows)
        values = values[order]
        values.flags.writeable = False
        families[name] = ([keys[i] for i in order], values)
    expected = assemble(keys, **families)
    actual = candidate(keys, families)
    for field in ('scores', 'labels'):
        np.testing.assert_array_equal(actual[field], expected[field])


def test_two_sums_preserve_underflow_that_single_weighted_sum_loses():
    tiny = np.nextafter(0., 1.)
    a = np.array([[tiny, 1., 0., 0., 0.]])
    c = np.array([[0., 1., 0., 0., 0.]])
    keys = ['synthetic']
    families = dict(vit=(keys, a), resnext=(keys, a), efficientnet=(keys, c), cropnet=(keys, c))
    expected = assemble(keys, **families)
    actual = candidate(keys, families)
    np.testing.assert_array_equal(actual['scores'], expected['scores'])
    assert actual['scores'][0, 0] == tiny
    shortcut = ordered_weighted_sum(np.stack([a[0], a[0], c[0], c[0]]), np.array([.5, .5, 1., 1.]))
    assert shortcut[0] == 0.


@pytest.mark.parametrize('fault', ['missing_identity', 'duplicate_identity', 'nonfinite', 'negative', 'unnormalized', 'wrong_width'])
def test_source_rejections_preserved(fault):
    keys = ['synthetic-a', 'synthetic-b']
    values = np.full((2, 5), .2)
    bad_keys = list(keys)
    if fault == 'missing_identity': bad_keys[0] = 'synthetic-other'
    elif fault == 'duplicate_identity': bad_keys[0] = bad_keys[1]
    elif fault == 'nonfinite': values[0, 0] = np.nan
    elif fault == 'negative': values[0, 0] = -.2
    elif fault == 'unnormalized': values[0, :] = .1
    else: values = values[:, :4]
    good = (keys, np.full((2, 5), .2))
    families = dict(vit=(bad_keys, values), resnext=good, efficientnet=good, cropnet=good)
    with pytest.raises(ValueError): assemble(keys, **families)
    with pytest.raises(ValueError): candidate(keys, families)

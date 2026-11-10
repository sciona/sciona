import numpy as np
import pytest

from sciona.bengali_routing import route_predictions
from sciona.confidence_routing import confidence_route_mask, fit_confidence_router


def test_float64_domain_adapter_preserves_existing_grapheme_routing():
    rng = np.random.default_rng(213)
    membership = rng.random(14784) < .2
    primary = rng.integers(0, 14784, 100)
    alternate = rng.integers(0, 14784, 100)
    confidence = np.linspace(0, 1, 100)
    for threshold in [0., .2, .5, 1.]:
        mask = confidence_route_mask(confidence, ~membership[alternate], threshold=threshold)
        expected = route_predictions(confidence, primary, alternate,
                                     seen_class_mask=membership, threshold=threshold)
        np.testing.assert_array_equal(np.where(mask, alternate, primary), expected)


@pytest.mark.parametrize('primary,alternate,confidence,eligible', [
    # Constructed manufacturing inspection: utility is correct fault classification.
    ([1., 0., 1.], [0., 1., 0.], [.8, .2, .1], [True, True, False]),
    # Constructed energy dispatch: utility is negative operating cost in common units.
    ([-10., -2., -5.], [-3., -8., -4.], [.2, .8, .3], [True, True, False]),
])
def test_two_distinct_synthetic_application_contracts(primary, alternate, confidence, eligible):
    result = fit_confidence_router(confidence, eligible, primary, alternate,
                                   sample_weight=[1., 1., 1.], thresholds=[1., .5, 0.])
    assert result['threshold'] == .5
    mask = confidence_route_mask(confidence, eligible, threshold=result['threshold'])
    expected = sum(a if use else p for a, p, use in zip(alternate, primary, mask))/3
    assert result['weighted_utility'] == pytest.approx(expected)


def test_weighted_grid_matches_independent_scalar_enumeration_and_tie_rule():
    rng = np.random.default_rng(83)
    for _ in range(20):
        confidence = rng.random(17)
        eligible = rng.random(17) > .3
        primary, alternate = rng.normal(size=(2, 17))
        weights = rng.integers(0, 20, 17)
        grid = [1., .5, .25, 0., .5]
        result = fit_confidence_router(confidence, eligible, primary, alternate,
                                       sample_weight=weights, thresholds=grid)
        expected = [sum(w*(a if c < t and e else p) for c,e,p,a,w in
                    zip(confidence,eligible,primary,alternate,weights))/sum(weights)
                    for t in sorted(set(grid))]
        np.testing.assert_allclose(result['candidate_utilities'], expected, rtol=1e-14, atol=1e-14)
        assert result['threshold'] == sorted(set(grid))[int(np.argmax(expected))]
    tied = fit_confidence_router([.5], [True], [1.], [1.], sample_weight=[1.], thresholds=[1., 0.])
    assert tied['threshold'] == 0.
    assert not confidence_route_mask([.5], [True], threshold=.5)[0]


@pytest.mark.parametrize('argument,value', [
    ('confidence', [np.nan]), ('alternate_eligible', [1]),
    ('primary_utility', [np.inf]), ('alternate_utility', [1., 2.]),
    ('sample_weight', [-1.]), ('sample_weight', [0.]),
    ('thresholds', []), ('thresholds', [True]), ('thresholds', [1.1]),
])
def test_invalid_calibration_contracts(argument, value):
    options = dict(confidence=[.5], alternate_eligible=[True], primary_utility=[0.],
                   alternate_utility=[1.], sample_weight=[1.], thresholds=[0., 1.])
    options[argument] = value
    with pytest.raises(ValueError):
        fit_confidence_router(**options)


def test_large_finite_weights_do_not_overflow():
    result = fit_confidence_router([.2, .8], [True, True], [0., 1.], [1., 0.],
                                   sample_weight=[1e308, 1e308], thresholds=[0., .5, 1.])
    assert result['threshold'] == .5 and result['weighted_utility'] == 1.

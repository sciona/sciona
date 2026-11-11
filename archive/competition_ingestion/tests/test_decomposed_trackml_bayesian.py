"""Synthetic checks of scoring diagnostics, isolation and qualified boundaries."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sciona.atoms.particle_tracking.track_matching.atoms import bayesian_neighbor_evaluation
from sciona.decomposed_trackml_bayesian_lifecycle import scorer_class
from sciona.trackml_bayesian import BayesianNeighborScorer
from scripts.review_trackml_bayesian_reuse import source_method


def inputs(n=8):
    rng = np.random.RandomState(385)
    xyz = rng.uniform(1., 4., (3, n))
    frame = pd.DataFrame(dict(extend_hit_id=np.arange(n), extend_index=np.arange(n),
        xn=xyz[0] + .1, yn=xyz[1] - .2, zn=xyz[2] + .3,
        cyl_closer=np.zeros(n, dtype=bool), next_id=np.zeros(n, dtype=int),
        hel_s=np.full(n, 2.)))
    run = SimpleNamespace(event=SimpleNamespace(hitCoordinatesById=lambda ids: xyz),
        neighbors=SimpleNamespace(evaluateLayerFunctions=lambda *a, **k:
            (np.full(n, .5), np.full(n, .6), np.full(n, .1), np.full(n, .2))),
        params={'nb__cut_factor': 1., 'nb__dist_trust': .8})
    return run, frame


@pytest.mark.parametrize('n', [0, 1, 8])
@pytest.mark.parametrize('thresholds', [(1., .8), (3., .5)])
def test_source_flags_and_every_diagnostic(n, thresholds):
    source, _ = source_method()
    run, frame = inputs(n)
    run.params.update(zip(('nb__cut_factor', 'nb__dist_trust'), thresholds))
    calls = []
    def provider(*args, **kwargs):
        calls.append(kwargs)
        return bayesian_neighbor_evaluation(*args, **kwargs)
    scorer = scorer_class(BayesianNeighborScorer, provider)()
    actual_details, expected_details = SimpleNamespace(), SimpleNamespace()
    actual = scorer.bayesianNeighborEvaluation(run, frame, details=actual_details)
    expected = source(BayesianNeighborScorer(), run, frame, details=expected_details)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert vars(actual_details).keys() == vars(expected_details).keys()
    for key in vars(expected_details):
        np.testing.assert_array_equal(getattr(actual_details, key), getattr(expected_details, key))
    assert calls == ([dict(cut_factor=thresholds[0], dist_trust=thresholds[1])] if n else [])


def test_invocation_isolation_and_provider_controls_flags():
    run, frame = inputs()
    original = BayesianNeighborScorer.bayesianNeighborEvaluation
    first = scorer_class(BayesianNeighborScorer, lambda *a, **k: ('first', 'flags'))()
    second = scorer_class(BayesianNeighborScorer, lambda *a, **k: ('second', 'flags'))()
    assert first.bayesianNeighborEvaluation(run, frame) == ('first', 'flags')
    assert second.bayesianNeighborEvaluation(run, frame) == ('second', 'flags')
    assert first.bayesianNeighborEvaluation(run, frame) == ('first', 'flags')
    assert BayesianNeighborScorer.bayesianNeighborEvaluation is original


@pytest.mark.parametrize('value', [0., -1., np.inf, np.nan, True])
@pytest.mark.parametrize('key', ['nb__cut_factor', 'nb__dist_trust'])
def test_invalid_threshold_rejected_before_provider(key, value):
    run, frame = inputs()
    run.params[key] = value
    def forbidden(*a, **k):
        pytest.fail('Provider must not receive invalid thresholds')
    with pytest.raises(ValueError, match='thresholds'):
        scorer_class(BayesianNeighborScorer, forbidden)().bayesianNeighborEvaluation(run, frame)


@pytest.mark.parametrize('kind', ['nan', 'zero_error', 'overflow', 'shape', 'dtype'])
def test_invalid_numeric_domain_rejected_before_provider(kind):
    run, frame = inputs()
    def forbidden(*a, **k):
        pytest.fail('Provider must not receive invalid vectors')
    scorer = scorer_class(BayesianNeighborScorer, forbidden)()
    if kind == 'nan':
        frame.loc[0, 'xn'] = np.nan
    elif kind == 'zero_error':
        scorer.predictRandomError = lambda *a, **k: (np.zeros(8), np.ones(8))
    elif kind == 'overflow':
        scorer.predictRandomError = lambda *a, **k: (np.full(8, 1e-300), np.ones(8))
    elif kind == 'shape':
        scorer.projectNeighborDisplacement = lambda *a: (np.ones(1), np.ones(8))
    else:
        scorer.projectNeighborDisplacement = lambda *a: (np.ones(8, dtype=np.float32), np.ones(8))
    with pytest.raises(ValueError):
        scorer.bayesianNeighborEvaluation(run, frame)

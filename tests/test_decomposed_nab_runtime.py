"""Synthetic causal equivalence and reusable threshold boundary checks."""
import copy
import json
import numpy as np
import pytest
from sciona.atoms.numerical.statistics.prior_threshold import prior_location_scale_threshold as threshold
from sciona.nab_contract import prepare, execute
from sciona import decomposed_nab_runtime as split


def payload(n=24, ddof=0):
    return dict(version=1, values=np.random.default_rng(51).normal(size=n).tolist(),
        detector=dict(window_size=3, fit_history=7, min_fit_windows=3,
            threshold_history=5, min_threshold_scores=3, n_estimators=5,
            max_samples=7, seed=42, std_ddof=ddof),
        windows=[[8, 10]] if n > 10 else [], probation_percent=.1,
        costs=dict(tpWeight=1., fpWeight=.11, fnWeight=1.))


def run(p):
    prepared = prepare(p)
    forest = split.forest_scores(prepared)
    thresholds = threshold(**split.threshold_inputs(prepared, forest))
    return split.result(prepared, split.detection(prepared, forest, thresholds))


@pytest.mark.parametrize('n,ddof', [(1,0), (5,1), (6,0), (8,1), (24,0), (24,1)])
def test_full_reconstruction_exact(n, ddof):
    p = payload(n, ddof)
    before = copy.deepcopy(p)
    assert run(p) == execute(prepare(p))
    assert p == before
    json.dumps(run(p), allow_nan=False)


def test_forests_exclude_current_and_labels(monkeypatch):
    from sklearn.ensemble import IsolationForest
    original = IsolationForest.fit
    seen = []
    def fit(self, x, *args, **kwargs):
        assert self.n_jobs == 1
        seen.append(x.copy())
        return original(self, x, *args, **kwargs)
    monkeypatch.setattr(IsolationForest, 'fit', fit)
    p = payload(6)
    p['values'] = [0., 1., 2., 3., 4., 999.]
    first = split.forest_scores(prepare(p))
    assert len(seen) == 1
    assert np.array_equal(seen[0], [[0,1,2], [1,2,3], [2,3,4]])
    p['windows'] = [[1,2]]
    assert split.forest_scores(prepare(p)) == first


def test_prefix_and_future_invariance():
    p = payload()
    full = run(p)['detection']
    changed = copy.deepcopy(p)
    changed['values'][18:] = [1000.]*6
    future = run(changed)['detection']
    p['values'] = p['values'][:18]
    prefix = run(p)['detection']
    for key in ('anomaly_scores', 'thresholds', 'detections'):
        assert prefix[key] == full[key][:18] == future[key][:18]


@pytest.mark.parametrize('multiplier', [-2., 0., .5, 3., 7.])
def test_threshold_independent_scalar_reference(multiplier):
    values = [1., -4., 9., 2., 20., -8.]
    before = list(values)
    actual = threshold(values, history=3, minimum=2, ddof=1, multiplier=multiplier)
    expected = [None, None]
    for i in range(2, len(values)):
        prior = values[max(0,i-3):i]
        mean = sum(prior)/len(prior)
        std = (sum((v-mean)**2 for v in prior)/(len(prior)-1))**.5
        expected.append(mean+multiplier*std)
    assert actual[:2] == expected[:2]
    # Independent Python sqrt and NumPy std may differ by a few ulps.
    assert actual[2:] == pytest.approx(expected[2:], rel=2e-15, abs=2e-15)
    assert values == before


def test_threshold_empty_singleton_and_strict_causality():
    assert threshold([], 1, 1, 0, 3.) == []
    assert threshold([5., 99., -5.], 1, 1, 0, 3.) == [None, 5., 99.]
    assert threshold([5.], 1, 1, 0, 3.) == [None]


@pytest.mark.parametrize('kwargs', [
    {'values':[True]}, {'values':[float('nan')]}, {'values':[float('inf')]},
    {'values':[10**400]}, {'values':['3']}, {'values':(1.,2.)},
    {'history':True}, {'history':0}, {'minimum':4}, {'minimum':0},
    {'ddof':2}, {'ddof':-1}, {'multiplier':True}, {'multiplier':float('inf')},
    {'values':[1e308,1e308,0.]},
])
def test_threshold_rejects_invalid_or_overflow(kwargs):
    controls=dict(values=[1.,2.,3.], history=3, minimum=2, ddof=0, multiplier=3.)
    controls.update(kwargs)
    with pytest.raises(ValueError):
        threshold(**controls)


@pytest.mark.parametrize('fault', ['missing_score', 'wrong_fits', 'interior_null', 'wrong_threshold_count', 'wrong_flags'])
def test_incomplete_intermediates_rejected(fault):
    prepared = prepare(payload(12))
    forest = split.forest_scores(prepared)
    thresholds = threshold(**split.threshold_inputs(prepared, forest))
    detected = split.detection(prepared, forest, thresholds)
    with pytest.raises(ValueError):
        if fault == 'missing_score':
            forest['anomaly_scores'].pop()
            split.threshold_inputs(prepared, forest)
        elif fault == 'wrong_fits':
            forest['models_fitted'] -= 1
            split.threshold_inputs(prepared, forest)
        elif fault == 'interior_null':
            forest['anomaly_scores'][-1] = None
            split.threshold_inputs(prepared, forest)
        elif fault == 'wrong_threshold_count':
            split.detection(prepared, forest, thresholds[:-1])
        else:
            detected['detections'][-1] = not detected['detections'][-1]
            split.result(prepared, detected)

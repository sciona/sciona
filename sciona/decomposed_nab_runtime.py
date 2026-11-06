"""Explicit forest, threshold policy and evaluation stages for the NAB intake.

The original full reconstruction remains the comparison oracle. This module
does not change its detector choice, fit population, scoring or normalization.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sciona.nab_contract import Prepared, prepare
from sciona.nab_scoring import evaluate


def checked(prepared):
    if not isinstance(prepared, Prepared):
        raise ValueError('Prepared NAB payload required')
    return prepare(prepared.payload).payload


def forest_scores(prepared):
    p = checked(prepared)
    x = np.asarray(p['values'])
    config = p['detector']
    # Validate the actual observation range as the historical detector does.
    if (x.dtype.kind not in 'fi' or not np.isfinite(x).all()
            or np.max(abs(x)) > np.finfo(np.float32).max):
        raise ValueError('Observations exceed forest numeric range')
    window = config['window_size']
    scores = [None] * len(x)
    fits = 0
    for i in range(window-1, len(x)):
        first = max(window-1, i-config['fit_history'])
        if i-first < config['min_fit_windows']:
            continue
        training = np.stack([x[j-window+1:j+1] for j in range(first, i)])
        model = IsolationForest(n_estimators=config['n_estimators'],
            max_samples=min(config['max_samples'], len(training)),
            contamination='auto', max_features=1., bootstrap=False, n_jobs=1,
            random_state=(config['seed']+i) % (2**32))
        model.fit(training)
        fits += 1
        score = float(-model.score_samples(x[i-window+1:i+1][None, :])[0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid forest anomaly score')
        scores[i] = score
    return dict(anomaly_scores=scores, models_fitted=fits)


def checked_scores(prepared, forest):
    p = checked(prepared)
    if not isinstance(forest, dict) or set(forest) != {'anomaly_scores', 'models_fitted'}:
        raise ValueError('Exact forest result required')
    scores = forest['anomaly_scores']
    warmup = min(len(p['values']), p['detector']['window_size']-1+p['detector']['min_fit_windows'])
    if (not isinstance(scores, list) or len(scores) != len(p['values'])
            or scores[:warmup] != [None]*warmup
            or any(type(v) is not float or not np.isfinite(v) or not 0 <= v <= 1
                   for v in scores[warmup:])
            or type(forest['models_fitted']) is not int
            or forest['models_fitted'] != len(scores)-warmup):
        raise ValueError('Complete causal forest population required')
    return p, scores, warmup


def threshold_inputs(prepared, forest):
    p, scores, warmup = checked_scores(prepared, forest)
    c = p['detector']
    return dict(values=scores[warmup:], history=c['threshold_history'],
                minimum=c['min_threshold_scores'], ddof=c['std_ddof'], multiplier=3.)


def detection(prepared, forest, thresholds):
    p, scores, warmup = checked_scores(prepared, forest)
    minimum = p['detector']['min_threshold_scores']
    if (not isinstance(thresholds, list) or len(thresholds) != len(scores)-warmup
            or thresholds[:minimum] != [None]*min(minimum, len(thresholds))
            or any(type(v) is not float or not np.isfinite(v) for v in thresholds[minimum:])):
        raise ValueError('Complete prior-only threshold population required')
    full = [None]*warmup + thresholds
    flags = [threshold is not None and score > threshold
             for score, threshold in zip(scores, full)]
    return dict(anomaly_scores=list(scores), thresholds=full, detections=flags,
                models_fitted=forest['models_fitted'])


def result(prepared, detected):
    p = checked(prepared)
    if not isinstance(detected, dict) or set(detected) != {
            'anomaly_scores', 'thresholds', 'detections', 'models_fitted'}:
        raise ValueError('Complete detection result required')
    forest = {k: detected[k] for k in ('anomaly_scores', 'models_fitted')}
    _, _, warmup = checked_scores(prepared, forest)
    thresholds = detected['thresholds']
    if (not isinstance(thresholds, list) or len(thresholds) != len(p['values'])
            or thresholds[:warmup] != [None]*warmup
            or not isinstance(detected['detections'], list)
            or any(type(v) is not bool for v in detected['detections'])
            or detection(prepared, forest, thresholds[warmup:]) != detected):
        raise ValueError('Detection controls or output population differ')
    scoring = evaluate([float(flag) for flag in detected['detections']], p['windows'],
        threshold=.5, probation_percent=p['probation_percent'], costs=p['costs'])
    return dict(detection=detected, evaluation=scoring)

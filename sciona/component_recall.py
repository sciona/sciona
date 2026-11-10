"""Additive utilities for weighted macro recall over aligned label components.

Only classes with positive validation weight enter each component's macro
average. The arithmetic mean of returned row utilities equals the weighted
component-macro score. Sample weights are already folded into the utilities;
do not apply them a second time. Labels and component importance are supplied
by domain adapters, without a built-in class vocabulary.
"""
import numpy as np


def component_recall_utilities(truth, predictions, *, sample_weight, component_weight):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if (truth.ndim != 2 or min(truth.shape) < 1 or predictions.shape != truth.shape
            or truth.dtype.kind not in 'iu' or predictions.dtype.kind not in 'iu'
            or (truth < 0).any() or (predictions < 0).any()):
        raise ValueError('Aligned nonnegative integer N-by-K component labels required')
    weights, importance = np.asarray(sample_weight), np.asarray(component_weight)
    for vector, size in ((weights, len(truth)), (importance, truth.shape[1])):
        if (vector.shape != (size,) or vector.dtype.kind not in 'fiu'
                or not np.isfinite(vector).all() or (vector < 0).any() or vector.max() <= 0):
            raise ValueError('Finite nonnegative aligned weights with positive total required')
    weights = weights.astype(np.float64)
    importance = importance.astype(np.float64)
    if not np.isfinite(weights).all() or not np.isfinite(importance).all():
        raise ValueError('Weights must be representable as finite float64')
    importance /= importance.max()
    importance /= importance.sum()
    utilities = np.zeros(len(truth), dtype=np.float64)
    for column, scale in enumerate(importance):
        classes = np.unique(truth[weights > 0, column])
        for label in classes:
            member = truth[:, column] == label
            correct = predictions[member, column] == label
            local = weights[member]/weights[member].max()
            fraction = local/local.sum()
            utilities[member] += len(truth)*scale/len(classes)*fraction*correct
    return utilities

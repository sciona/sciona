"""Draft Otto source-policy adapter over reusable categorical operations.

The retained historical implementation is unchanged. This candidate adapter
requires separate graph/provider qualification before catalog promotion.
It is outside the historical otto_* module inventory and is not imported by
any served provider; its code must be included in the new qualification.
"""
import numpy as np
from sciona.probability_ensemble import (
    mean_probability_bag, geometric_arithmetic_probability_blend,
)


def blend(xgboost, neural, adaboost_extratrees):
    means = []
    for values, runs in ((xgboost, 250), (neural, 600), (adaboost_extratrees, 250)):
        array = np.asarray(values)
        if array.ndim != 3 or array.shape[0] != runs or array.shape[2] != 9:
            raise ValueError('Full source bag counts and nine classes required')
        means.append(mean_probability_bag(array))
    return geometric_arithmetic_probability_blend(*means, .85, .15, .65)

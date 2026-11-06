"""Draft category feature policy for the complete tabular recipe.

Count/frequency/target statistics use training rows only. Target features for
training rows exclude the entire supplied group; query rows use full-training
statistics. This explicit split policy is separate from the reused numerical
providers. Callers normalize categories and enforce outer population isolation.
"""
from collections import Counter
from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np

from sciona.atoms.ml.tabular.gradient_boosting.atoms import (
    frequency_encode_fit, frequency_encode, target_encode,
)


def _table(rows, width=None):
    if type(rows) is not list or not rows or any(type(row) is not list for row in rows):
        raise ValueError('Nonempty list of category rows required')
    widths = {len(row) for row in rows}
    if len(widths) != 1 or not len(rows[0]) or (width is not None and len(rows[0]) != width):
        raise ValueError('Fixed nonzero category width required')
    if any(type(value) is not str for row in rows for value in row):
        raise ValueError('Explicit string category tokens required')
    return np.asarray(rows, dtype=object)


def _interactions(pairs, width):
    if type(pairs) is not list:
        raise ValueError('Explicit list of category interaction pairs required')
    result = []
    for pair in pairs:
        if type(pair) not in (list, tuple) or len(pair) != 2 or any(type(i) is not int or not 0 <= i < width for i in pair):
            raise ValueError('Valid category column indices required')
        left, right = pair
        if left == right or tuple(sorted(pair)) in {tuple(sorted(p)) for p in result}:
            raise ValueError('Distinct unique interaction columns required')
        result.append((left, right))
    return tuple(result)


def _columns(values, interactions):
    columns = [values[:, i] for i in range(values.shape[1])]
    for left, right in interactions:
        # JSON preserves the ordered pair without delimiter collisions.
        columns.append(np.array([json.dumps([a, b], ensure_ascii=True, separators=(',', ':'))
            for a, b in zip(values[:, left], values[:, right], strict=True)], dtype=object))
    return columns


@dataclass(frozen=True)
class CategoryFeatures:
    width: int
    interactions: tuple
    counts: tuple
    frequencies: tuple
    target_means: tuple
    prior: float

    def transform(self, rows):
        """Apply frozen maps in row order; unknowns use 0 count/frequency and prior."""
        columns = _columns(_table(rows, self.width), self.interactions)
        pieces = []
        for column, counts, frequencies, targets in zip(columns, self.counts, self.frequencies, self.target_means, strict=True):
            pieces.extend((np.array([counts.get(v, 0) for v in column], dtype=np.float64),
                frequency_encode(column, frequencies, unknown_value=0.),
                np.array([targets.get(v, self.prior) for v in column], dtype=np.float64)))
        result = np.column_stack(pieces)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite category features')
        return result


def fit_category_features(rows, targets, groups, *, interactions, smoothing):
    """Return frozen inference state and group-excluded training features.

Targets are arbitrary finite real values, not necessarily binary or labels
from a particular discipline. Smoothing is nonnegative and explicit. Each
column (originals then declared interactions) contributes count, frequency,
target mean in that order. Training target means and fallback prior exclude
the row's entire group, while unsupervised counts use all supplied training
rows. At least two groups are required. Group exclusion repeats fitting and
is intended for bounded dense tables; it is not a large-table runtime claim.
"""
    values = _table(rows)
    pairs = _interactions(interactions, values.shape[1])
    if type(targets) is not list or len(targets) != len(rows) or any(type(v) not in (int, float) for v in targets):
        raise ValueError('Aligned real-valued targets required')
    y = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Finite targets required')
    if type(groups) is not list or len(groups) != len(rows) or any(type(g) is not str or not g for g in groups) or len(set(groups)) < 2:
        raise ValueError('Aligned groups with at least two distinct groups required')
    if type(smoothing) not in (int, float) or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError('Finite nonnegative smoothing required')
    columns = _columns(values, pairs)
    memberships = np.asarray(groups, dtype=object)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            prior = float(np.mean(y))
            state = CategoryFeatures(values.shape[1], pairs,
                tuple(MappingProxyType(dict(Counter(c))) for c in columns),
                tuple(MappingProxyType(frequency_encode_fit(c)) for c in columns),
                tuple(MappingProxyType(target_encode(c, y, smoothing=smoothing, prior=prior)) for c in columns), prior)
            training = state.transform(rows)
            for group in sorted(set(groups)):
                held = memberships == group
                fitted = ~held
                group_prior = float(np.mean(y[fitted]))
                for index, column in enumerate(columns):
                    mapping = target_encode(column[fitted], y[fitted], smoothing=smoothing, prior=group_prior)
                    training[held, 3*index+2] = [mapping.get(v, group_prior) for v in column[held]]
    except FloatingPointError as error:
        raise ValueError('Unrepresentable category statistic') from error
    if not np.isfinite(training).all():
        raise ValueError('Nonfinite group-excluded category features')
    return state, training

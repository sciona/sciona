"""Pending binary tabular recipe features; catalog qualification is separate."""
from dataclasses import dataclass
import math

import numpy as np

from sciona.tabular_ensemble_features import TabularFeatures, validate_table
from sciona.tabular_ensemble_training import binary_labels, group_ids
from sciona.tabular_category_features import fit_category_features, _interactions


def _logs(numeric, columns):
    return [list(row)+[None if row[i] is None else math.copysign(math.log1p(abs(float(row[i]))), float(row[i]))
        for i in columns] for row in numeric]


def _mode_fraction(column):
    return np.unique(column, return_counts=True)[1].max()/len(column)


@dataclass(frozen=True)
class CompleteTabularFeatures:
    widths: tuple
    log_columns: tuple
    categorical_columns: tuple
    base: object
    base_columns: tuple
    categories: object

    def transform(self, numeric, categorical):
        """Frozen query transform, never the target-encoded training transform."""
        numbers, tokens = validate_table(numeric, categorical)
        if (numbers.shape[1], tokens.shape[1]) != self.widths:
            raise ValueError('Original table widths differ')
        selected = [[row[i] for i in self.categorical_columns] for row in categorical]
        base = self.base.transform(_logs(numeric, self.log_columns), selected)[:, self.base_columns]
        if self.categories is None:
            result = base
        else:
            result = np.column_stack((base, self.categories.transform(tokens[:, self.categorical_columns].tolist())))
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite complete table features')
        return result


def fit_binary_tabular_features(numeric, categorical, labels, groups, *, clip_low, clip_high,
                               max_mode_fraction, log_columns, interactions, smoothing):
    """Return frozen state and the distinct group-excluded training matrix.

Cleaning fits only supplied training rows. Drop constant, duplicate and
quasi-constant category columns; numeric duplicate/constant screening occurs
after median imputation and quantile clipping. Drop numeric columns whose
modal share exceeds max_mode_fraction. Signed log1p copies of explicitly
selected numeric columns are appended before numeric cleaning. Units and
choice of logarithm remain recipe policy. Category interactions use original
column indices; interactions involving screened columns are screened too.

Keep one-hot features, then count/frequency/group-excluded target features.
Do not call transform on training rows to fit a model: that would replace
excluded target statistics with full-training maps. Caller enforces outer
folds and population isolation. Dense state and arrays are execution-local.
"""
    numbers, tokens = validate_table(numeric, categorical)
    binary_labels(labels, len(numbers)); group_ids(groups, len(numbers))
    if type(max_mode_fraction) not in (int, float) or not 0 < max_mode_fraction <= 1:
        raise ValueError('Modal fraction must be in (0,1]')
    if type(log_columns) is not list or any(type(i) is not int or not 0 <= i < numbers.shape[1] for i in log_columns) or len(set(log_columns)) != len(log_columns):
        raise ValueError('Distinct valid log-transform columns required')
    pairs = _interactions(interactions, tokens.shape[1])
    if type(smoothing) not in (int, float) or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError('Finite nonnegative smoothing required')
    kept = []
    for i in range(tokens.shape[1]):
        column = tokens[:, i]
        if len(set(column)) == 1 or _mode_fraction(column) > max_mode_fraction:
            continue
        if any(np.array_equal(column, tokens[:, j]) for j in kept):
            continue
        kept.append(i)
    selected = [[row[i] for i in kept] for row in categorical]
    expanded = _logs(numeric, log_columns)
    base = TabularFeatures().fit(expanded, selected, clip_low=clip_low, clip_high=clip_high)
    original = base.transform(expanded, selected)
    numeric_width = len(base.kept)
    # The old base appends frequency columns. Replace them with the new triplets.
    base_columns = tuple([i for i in range(numeric_width) if _mode_fraction(original[:, i]) <= max_mode_fraction]
        + list(range(numeric_width, original.shape[1]-len(kept))))
    categorical_state = None
    training = original[:, base_columns]
    if kept:
        remapped = [[kept.index(a), kept.index(b)] for a, b in pairs if a in kept and b in kept]
        categorical_state, encoded_training = fit_category_features(tokens[:, kept].tolist(), labels, groups,
            interactions=remapped, smoothing=smoothing)
        training = np.column_stack((training, encoded_training))
    if training.shape[1] == 0 or not np.isfinite(training).all():
        raise ValueError('No finite varying table features remain')
    state = CompleteTabularFeatures((numbers.shape[1], tokens.shape[1]), tuple(log_columns), tuple(kept),
        base, base_columns, categorical_state)
    return state, training

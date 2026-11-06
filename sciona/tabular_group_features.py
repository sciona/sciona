"""Training-only group statistics for pending tabular recipes.

Keys are caller-selected feature keys, not validation-fold identifiers. The
caller supplies finite numeric values after its chosen missing-value policy.
This module does not inspect labels or infer entity, session, or time semantics.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from sciona.atoms.ml.tabular.gradient_boosting.atoms import group_aggregate


_STATISTICS = frozenset(('count', 'sum', 'mean', 'var', 'std', 'min', 'max'))


def _keys(keys):
    if type(keys) is not list or not keys or any(type(key) is not str for key in keys):
        raise ValueError('Nonempty list of string feature keys required')
    return keys


@dataclass(frozen=True)
class GroupFeatures:
    """Frozen key-to-feature lookup; unseen keys use the explicit fill value."""

    mapping: object
    width: int
    unknown_value: float

    def transform(self, keys):
        keys = _keys(keys)
        fallback = (self.unknown_value,) * self.width
        return np.asarray([self.mapping.get(key, fallback) for key in keys], dtype=np.float64)


def fit_group_features(values, keys, *, statistics, unknown_value):
    """Return frozen state and aligned training features.

Output order is numeric column first, then requested statistic. Variance and
standard deviation use population denominators. Count is repeated per numeric
column for a consistent rectangular contract. Only supplied training values
contribute; querying a new batch never updates the maps. Temporal/as-of or
leave-group-out aggregation requires a separate, explicit recipe policy.
"""
    keys = _keys(keys)
    if (type(values) is not list or len(values) != len(keys)
            or any(type(row) is not list or not row for row in values)
            or len({len(row) for row in values}) != 1
            or any(type(value) not in (int, float) for row in values for value in row)):
        raise ValueError('Aligned nonempty rectangular numeric rows required')
    try:
        numbers = np.asarray(values, dtype=np.float64)
    except (OverflowError, ValueError) as exc:
        raise ValueError('Values must be representable as finite float64') from exc
    if not np.isfinite(numbers).all():
        raise ValueError('Finite numeric values required')
    if (type(statistics) is not list or not statistics
            or any(type(stat) is not str or stat not in _STATISTICS for stat in statistics)
            or len(set(statistics)) != len(statistics)):
        raise ValueError('Distinct supported statistics required')
    if type(unknown_value) not in (int, float):
        raise ValueError('Explicit finite unknown-key value required')
    try:
        fill = float(unknown_value)
    except OverflowError as exc:
        raise ValueError('Unknown-key value must be representable as float64') from exc
    if not np.isfinite(fill):
        raise ValueError('Explicit finite unknown-key value required')
    key_array = np.asarray(keys, dtype=object)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        columns = [group_aggregate(numbers[:, column], key_array, stat)
                   for column in range(numbers.shape[1]) for stat in statistics]
    aggregates = np.column_stack(columns)
    if not np.isfinite(aggregates).all():
        raise ValueError('Nonfinite group aggregate')
    mapping = MappingProxyType({key: tuple(row) for key, row in zip(np.unique(key_array), aggregates)})
    state = GroupFeatures(mapping, aggregates.shape[1], fill)
    return state, state.transform(keys)

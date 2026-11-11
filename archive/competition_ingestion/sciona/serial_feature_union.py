"""Serial feature branches with explicit input columns and stable output order.

Callers provide (name, ordered_columns, transformer) triples. Each transformer
must preserve row order and implement fit_transform and transform. This core
creates no workers; transformer implementations must independently honor the
caller's shared resource budget. Object-valued table cells must be immutable
scalars. No application field names, feature units or response transforms are
inferred. Fitted transformer objects are runtime state, not portable evidence.
"""
from copy import deepcopy

import pandas as pd
from pandas.api.types import is_scalar
from scipy import sparse


class SerialFeatureUnion:
    def __init__(self, branches):
        self.branches = branches
        self._fitted = None

    @staticmethod
    def _specifications(branches):
        if type(branches) is not list or not branches:
            raise ValueError('Nonempty ordered branch list required')
        names = set()
        for branch in branches:
            if type(branch) not in (tuple, list) or len(branch) != 3:
                raise ValueError('Each branch requires name, columns and transformer')
            name, columns, transformer = branch
            if type(name) is not str or not name.strip() or name in names:
                raise ValueError('Unique nonblank branch names required')
            names.add(name)
            if (type(columns) is not list or not columns or any(type(c) is not str for c in columns)
                    or len(set(columns)) != len(columns)):
                raise ValueError('Explicit ordered unique input columns required')
            if not callable(getattr(transformer, 'fit_transform', None)) or not callable(getattr(transformer, 'transform', None)):
                raise ValueError('Fit/transform feature interface required')

    @staticmethod
    def _input(frame, columns):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError('DataFrame with unique columns required')
        if any(column not in frame.columns for column in columns):
            raise ValueError('Required branch column missing')
        selected = frame.loc[:, columns].copy(deep=True)
        for column in selected:
            if selected[column].dtype == object and not all(is_scalar(v) for v in selected[column]):
                raise ValueError('Object cells must be immutable scalar values')
        return selected

    @staticmethod
    def _output(values, rows):
        if not hasattr(values, 'dtype') or values.dtype.kind not in 'biuf' or values.ndim != 2:
            raise ValueError('Two-dimensional real numeric branch output required')
        if values.shape[0] != rows or values.shape[1] < 1:
            raise ValueError('Branch must preserve row count and produce features')
        return sparse.csr_matrix(values)

    def fit_transform(self, frame, targets=None):
        # A failed refit invalidates old state instead of leaving a partial fit.
        self._fitted = None
        self._specifications(self.branches)
        if targets is not None and len(targets) != len(frame):
            raise ValueError('Targets must align to input rows')
        fitted, results = [], []
        for name, columns, transformer in self.branches:
            current = deepcopy(transformer)
            values = current.fit_transform(self._input(frame, columns), deepcopy(targets))
            output = self._output(values, len(frame))
            fitted.append((name, list(columns), current, output.shape[1]))
            results.append(output)
        result = sparse.hstack(results).tocsr()
        self._fitted = fitted
        return result

    def transform(self, frame):
        if self._fitted is None:
            raise ValueError('Complete fitted branches required')
        results = []
        for _, columns, transformer, width in self._fitted:
            output = self._output(transformer.transform(self._input(frame, columns)), len(frame))
            if output.shape[1] != width:
                raise ValueError('Fitted feature width changed')
            results.append(output)
        return sparse.hstack(results).tocsr()

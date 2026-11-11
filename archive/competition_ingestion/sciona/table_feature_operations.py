"""Explicit reusable table adapters with JSON-only configuration state.

No source field names, separator tokens or missing-value policies are implied.
These stateless operations preserve input row order. Object cells must be
immutable scalars; text concatenation requires strings rather than silently
coercing missing or numeric values. Caller owns feature semantics and units.
"""
import json
from importlib.metadata import version

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


def _columns(columns):
    if (type(columns) is not list or not columns or any(type(c) is not str for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError('Nonempty ordered unique string column names required')


def _frame(frame):
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or any(type(c) is not str for c in frame.columns)):
        raise ValueError('DataFrame with unique string column names required')
    for name in frame:
        if frame[name].dtype == object and not all(is_scalar(v) for v in frame[name]):
            raise ValueError('Object cells must be immutable scalars')


class _Stateless(TransformerMixin, BaseEstimator):
    def fit(self, values, y=None):
        self._configuration()
        return self

    def __sklearn_is_fitted__(self):
        return True


class TableColumnSelector(_Stateless):
    def __init__(self, columns, return_vector=True, order='declared'):
        self.columns = columns
        self.return_vector = return_vector
        self.order = order

    def _configuration(self):
        _columns(self.columns)
        if type(self.return_vector) is not bool or self.order not in ('declared', 'frame'):
            raise ValueError('Explicit selection order and boolean vector policy required')

    def transform(self, frame):
        self._configuration()
        _frame(frame)
        if any(c not in frame for c in self.columns):
            raise ValueError('Required column missing')
        columns = self.columns if self.order == 'declared' else [c for c in frame if c in self.columns]
        if len(columns) == 1 and self.return_vector:
            return list(frame[columns[0]])
        return frame.loc[:, columns].copy(deep=True)


class TableTextConcat(_Stateless):
    def __init__(self, columns, prefixes, output_column):
        self.columns = columns
        self.prefixes = prefixes
        self.output_column = output_column

    def _configuration(self):
        _columns(self.columns)
        if (type(self.prefixes) is not list or len(self.prefixes) != len(self.columns)
                or any(type(p) is not str for p in self.prefixes)
                or type(self.output_column) is not str or self.output_column in self.columns):
            raise ValueError('Aligned explicit prefixes and distinct output column required')

    def transform(self, frame):
        self._configuration()
        _frame(frame)
        if any(c not in frame for c in self.columns):
            raise ValueError('Required text column missing')
        if any(not all(isinstance(v, str) for v in frame[c]) for c in self.columns):
            raise ValueError('Resolve missing values and text coercion upstream')
        result = frame.copy(deep=True)
        result[self.output_column] = ''
        for column, prefix in zip(self.columns, self.prefixes):
            result[self.output_column] += prefix
            result[self.output_column] += frame[column]
        return result


class TableToRecords(_Stateless):
    def _configuration(self):
        pass

    def transform(self, frame):
        _frame(frame)
        return frame.to_dict(orient='records')


class NumericCSR(_Stateless):
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def _configuration(self):
        if self.dtype not in ('float32', 'float64'):
            raise ValueError('Explicit float32 or float64 output required')

    def transform(self, values):
        self._configuration()
        if not hasattr(values, 'dtype'):
            values = np.asarray(values)
        if values.ndim != 2 or values.dtype.kind not in 'biuf':
            raise ValueError('Two-dimensional real numeric input required')
        return sparse.csr_matrix(values, dtype=np.dtype(self.dtype), copy=True)


_KINDS = {'select_columns': TableColumnSelector, 'concat_text': TableTextConcat,
          'table_records': TableToRecords, 'numeric_csr': NumericCSR}


def _runtime():
    return {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')}


def export_table_operation(operation):
    kind = next((name for name, cls in _KINDS.items() if type(operation) is cls), None)
    if kind is None:
        raise ValueError('Known table operation required')
    operation._configuration()
    return dict(format='sciona.table_operation.v1', kind=kind, runtime=_runtime(),
                parameters=json.loads(json.dumps(operation.get_params(deep=False), allow_nan=False)))


def restore_table_operation(state):
    if (type(state) is not dict or set(state) != {'format', 'kind', 'runtime', 'parameters'}
            or state['format'] != 'sciona.table_operation.v1'
            or type(state['kind']) is not str or state['kind'] not in _KINDS
            or state['runtime'] != _runtime() or type(state['parameters']) is not dict):
        raise ValueError('Known exact-runtime table operation required')
    params = json.loads(json.dumps(state['parameters'], allow_nan=False))
    # Only known constructors are allowed; no import paths or executable state.
    import inspect
    cls = _KINDS[state['kind']]
    if set(params) != set(inspect.signature(cls).parameters):
        raise ValueError('Exact operation parameters required')
    result = cls(**params)
    result._configuration()
    return result

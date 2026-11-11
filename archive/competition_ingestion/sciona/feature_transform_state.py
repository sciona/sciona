"""JSON prediction-state codec for explicitly supported feature compositions.

No callable, import path, pickle or arbitrary estimator is restored. Private
vocabulary and column identities remain runtime state. Restored unions are
prediction-only; their fitting templates are deliberately absent. Callers own
row-order semantics, preprocessing and equivalence of feature coordinates.
"""
from importlib.metadata import version

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sciona.serial_feature_union import SerialFeatureUnion
from sciona.text_vectorizer_state import export_text_vectorizer, restore_text_vectorizer
from sciona.dict_vectorizer_state import export_dict_vectorizer, restore_dict_vectorizer
from sciona.table_feature_operations import (
    TableColumnSelector, TableTextConcat, TableToRecords, NumericCSR,
    export_table_operation, restore_table_operation,
)


def _names(names):
    if (not names or any(type(n) is not str or not n.strip() or '__' in n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('Unique nonblank composition names without double underscores required')


def export_feature_transform(transformer, _depth=0):
    if _depth > 64:
        raise ValueError('Composition nesting limit exceeded')
    recurse = lambda child: export_feature_transform(child, _depth + 1)
    if type(transformer) is Pipeline:
        if transformer.memory is not None or transformer.transform_input is not None or transformer.verbose:
            raise ValueError('Only plain prediction pipelines are supported')
        _names([name for name, _ in transformer.steps])
        kind, payload = 'pipeline', [dict(name=name, transform=recurse(child)) for name, child in transformer.steps]
    elif type(transformer) is SerialFeatureUnion:
        if transformer._fitted is None:
            raise ValueError('Fitted serial union required')
        _names([row[0] for row in transformer._fitted])
        kind, payload = 'serial_union', [dict(name=name, columns=list(columns), width=width,
                                             transform=recurse(child))
                                        for name, columns, child, width in transformer._fitted]
    elif type(transformer) in (CountVectorizer, TfidfVectorizer):
        kind, payload = 'text_encoder', export_text_vectorizer(transformer)
    elif type(transformer) is DictVectorizer:
        kind, payload = 'dictionary_encoder', export_dict_vectorizer(transformer)
    elif type(transformer) in (TableColumnSelector, TableTextConcat, TableToRecords, NumericCSR):
        kind, payload = 'table_operation', export_table_operation(transformer)
    elif type(transformer) is FunctionTransformer and transformer.get_params() == FunctionTransformer().get_params():
        kind, payload = 'identity', None
    else:
        raise ValueError('Unsupported feature operation; no executable fallback')
    return dict(format='sciona.feature_transform_prediction.v1', sklearn_version=version('scikit-learn'),
                kind=kind, payload=payload)


def restore_feature_transform(state, _depth=0):
    if _depth > 64:
        raise ValueError('Composition nesting limit exceeded')
    if (type(state) is not dict or set(state) != {'format', 'sklearn_version', 'kind', 'payload'}
            or state['format'] != 'sciona.feature_transform_prediction.v1'
            or state['sklearn_version'] != version('scikit-learn')):
        raise ValueError('Known exact-version feature composition required')
    kind, payload = state['kind'], state['payload']
    if kind in ('pipeline', 'serial_union'):
        keys = {'name', 'transform'} if kind == 'pipeline' else {'name', 'columns', 'width', 'transform'}
        if type(payload) is not list or not payload or any(type(row) is not dict or set(row) != keys for row in payload):
            raise ValueError('Exact ordered composition records required')
        _names([row['name'] for row in payload])
        restored = []
        for row in payload:
            child = restore_feature_transform(row['transform'], _depth + 1)
            if kind == 'pipeline':
                restored.append((row['name'], child))
            else:
                columns = row['columns']
                if (type(columns) is not list or not columns or any(type(c) is not str for c in columns)
                        or len(set(columns)) != len(columns) or type(row['width']) is not int or row['width'] < 1):
                    raise ValueError('Explicit branch columns and positive fitted width required')
                restored.append((row['name'], list(columns), child, row['width']))
        if kind == 'pipeline':
            result = Pipeline(restored)
            result._validate_steps()
            return result
        result = SerialFeatureUnion([])
        result._fitted = restored
        return result
    if kind == 'text_encoder':
        return restore_text_vectorizer(payload)
    if kind == 'dictionary_encoder':
        return restore_dict_vectorizer(payload)
    if kind == 'table_operation':
        return restore_table_operation(payload)
    if kind == 'identity' and payload is None:
        return FunctionTransformer()
    raise ValueError('Unsupported feature kind; no executable fallback')

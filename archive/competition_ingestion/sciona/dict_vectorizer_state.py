"""JSON prediction state for known sklearn dictionary feature encoders.

Fitted feature names are private runtime state. Only string feature names and
float32/float64 outputs are supported. Native category-name construction,
unknown-category handling and numeric-value semantics are preserved; callers
must avoid ambiguous separator-based names and keep preprocessing consistent.
"""
from importlib.metadata import version
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted


def _validate(state):
    fields = {'format', 'sklearn_version', 'dtype', 'separator', 'sparse', 'sort', 'feature_names'}
    if (type(state) is not dict or set(state) != fields
            or state['format'] != 'sciona.dict_vectorizer_prediction.v1'
            or state['sklearn_version'] != version('scikit-learn')
            or state['dtype'] not in ('float32', 'float64')
            or type(state['separator']) is not str
            or type(state['sparse']) is not bool or type(state['sort']) is not bool):
        raise ValueError('Known exact-version dictionary encoder state required')
    names = state['feature_names']
    if (type(names) is not list or not names or any(type(n) is not str for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('Nonempty unique ordered string feature names required')
    # Do not sort names here: their stored order is the learned coordinate.


def export_dict_vectorizer(model):
    if type(model) is not DictVectorizer:
        raise ValueError('Known dictionary encoder required')
    check_is_fitted(model, ['feature_names_', 'vocabulary_'])
    names = list(model.feature_names_)
    state = dict(format='sciona.dict_vectorizer_prediction.v1',
        sklearn_version=version('scikit-learn'), dtype=np.dtype(model.dtype).name,
        separator=model.separator, sparse=model.sparse, sort=model.sort, feature_names=names)
    _validate(state)
    if model.vocabulary_ != {name: i for i, name in enumerate(names)}:
        raise ValueError('Fitted vocabulary and feature order disagree')
    return state


def restore_dict_vectorizer(state):
    _validate(state)
    model = DictVectorizer(dtype=np.dtype(state['dtype']).type, separator=state['separator'],
                           sparse=state['sparse'], sort=state['sort'])
    model.feature_names_ = list(state['feature_names'])
    model.vocabulary_ = {name: i for i, name in enumerate(model.feature_names_)}
    return model

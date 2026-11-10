"""Strict numeric prediction state for the fixed OSIC quantile architecture.

No executable graph, Python object, callback or optimizer state is serialized.
Reconstruction uses the known local architecture. This format supports inference,
not training continuation. Coordinate names are explicit caller declarations.
"""
import numpy as np
from sciona.osic_quantile_model import build_model

FORMAT = 'sciona.osic_quantile_prediction.v1'


def _coordinates(columns, feature_coordinate, response_coordinate):
    if (type(columns) is not list or not columns
            or any(type(c) is not str or not c.strip() for c in columns)
            or len(set(columns)) != len(columns)
            or any(type(c) is not str or not c.strip() for c in (feature_coordinate, response_coordinate))):
        raise ValueError('Unique ordered features and explicit feature/response coordinates required')


def export_prediction_state(model, columns, feature_coordinate, response_coordinate):
    _coordinates(columns, feature_coordinate, response_coordinate)
    expected = build_model(len(columns))
    if (type(model) is not type(expected) or model.get_config() != expected.get_config()
            or int(model.optimizer.iterations.numpy()) < 1):
        raise ValueError('A trained instance of the exact known architecture is required')
    weights = model.get_weights()
    if (len(weights) != len(expected.get_weights()) or any(
            a.shape != b.shape or a.dtype != np.float32 or not np.isfinite(a).all()
            for a, b in zip(weights, expected.get_weights()))):
        raise ValueError('Complete finite float32 model weights required')
    return dict(format=FORMAT, feature_columns=list(columns), feature_coordinate=feature_coordinate,
                response_coordinate=response_coordinate, weights=[v.tolist() for v in weights])


def restore_prediction_state(state, columns, feature_coordinate, response_coordinate):
    _coordinates(columns, feature_coordinate, response_coordinate)
    fields = {'format', 'feature_columns', 'feature_coordinate', 'response_coordinate', 'weights'}
    if (type(state) is not dict or set(state) != fields or state['format'] != FORMAT
            or state['feature_columns'] != columns or state['feature_coordinate'] != feature_coordinate
            or state['response_coordinate'] != response_coordinate):
        raise ValueError('Exact known format and matching ordered coordinates required')
    shapes = [(len(columns), 100), (100,), (100, 100), (100,), (100, 3), (3,), (100, 3), (3,)]
    if type(state['weights']) is not list or len(state['weights']) != len(shapes):
        raise ValueError('Complete fixed weight inventory required')
    weights = []
    for raw, shape in zip(state['weights'], shapes):
        if type(raw) is not list:
            raise ValueError('Numeric JSON arrays required')
        objects = np.asarray(raw, dtype=object)
        if objects.shape != shape or any(type(x) not in (int, float) for x in objects.ravel()):
            raise ValueError('Exact numeric weight shapes required; booleans and strings are invalid')
        with np.errstate(over='ignore', invalid='ignore'):
            values = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Weights must be finite and float32-representable')
        weights.append(values)
    model = build_model(len(columns))
    if [v.shape for v in model.get_weights()] != shapes:
        raise ValueError('Local architecture changed')
    model.set_weights(weights)
    return model

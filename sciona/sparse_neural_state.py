"""Numeric JSON prediction state for the reusable sparse neural implementation.

No graphs, Python objects, optimizer slots or executable code are serialized.
Restoration constructs the known implementation and assigns validated numeric
weights. Coordinate identifiers are caller declarations, not semantic proof.
"""
from sciona.sparse_neural_regression import SparseRegressor, QuantileSparseRegressor, prelu, tf
from sciona.quantile_target_encoding import quantile_midpoints
import numpy as np


def _identities(columns, feature_coordinate, response_coordinate, width):
    if (type(columns) is not list or len(columns) != width or width < 1
            or any(type(c) is not str or not c.strip() for c in columns) or len(set(columns)) != width):
        raise ValueError('Unique ordered feature identities required')
    if any(type(c) is not str or not c.strip() for c in (feature_coordinate, response_coordinate)):
        raise ValueError('Explicit feature and response coordinate identifiers required')


def export_prediction_state(model, feature_columns, feature_coordinate, response_coordinate):
    if type(model) not in (SparseRegressor, QuantileSparseRegressor) or not model.is_fitted:
        raise ValueError('Open fitted reusable model required')
    _identities(feature_columns, feature_coordinate, response_coordinate, model.n_features_in_)
    if model.actfunc is prelu:
        activation = 'prelu'
    elif model.actfunc is tf.nn.relu:
        activation = 'relu'
    else:
        raise ValueError('Only known relu/prelu activations have a portable format')
    with model._graph.as_default():
        variables = tf.trainable_variables()
    values = model._session.run(variables)
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values):
        raise ValueError('Finite float32 learned variables required')
    quantile = type(model) is QuantileSparseRegressor
    decoder = dict(boundaries=model.percentiles.tolist()) if quantile else (
        dict(mean=float(model.target_scaler.mean_[0]), scale=float(model.target_scaler.scale_[0]))
        if model.use_target_scaling else None)
    return dict(format='sciona.sparse_neural_prediction.v1', mode='quantile' if quantile else 'mse',
        feature_columns=list(feature_columns), feature_coordinate=feature_coordinate,
        response_coordinate=response_coordinate, n_hidden=list(model.n_hidden), activation=activation,
        binary_X=model.binary_X, use_target_scaling=model.use_target_scaling, n_bins=model.n_bins,
        decoder=decoder, weights={v.name: a.tolist() for v, a in zip(variables, values)})


def restore_prediction_state(state, feature_columns, feature_coordinate, response_coordinate):
    fields = {'format', 'mode', 'feature_columns', 'feature_coordinate', 'response_coordinate',
              'n_hidden', 'activation', 'binary_X', 'use_target_scaling', 'n_bins', 'decoder', 'weights'}
    if (type(state) is not dict or set(state) != fields
            or state['format'] != 'sciona.sparse_neural_prediction.v1'
            or state['mode'] not in ('mse', 'quantile') or state['activation'] not in ('relu', 'prelu')):
        raise ValueError('Known exact prediction-state format required')
    _identities(feature_columns, feature_coordinate, response_coordinate,
                len(feature_columns) if type(feature_columns) is list else 0)
    if (state['feature_columns'] != feature_columns or state['feature_coordinate'] != feature_coordinate
            or state['response_coordinate'] != response_coordinate):
        raise ValueError('Feature order and coordinate identities must match fitted state')
    cls = QuantileSparseRegressor if state['mode'] == 'quantile' else SparseRegressor
    model = cls(n_hidden=state['n_hidden'], n_bins=state['n_bins'], binary_X=state['binary_X'],
                use_target_scaling=state['use_target_scaling'],
                actfunc=prelu if state['activation'] == 'prelu' else tf.nn.relu)
    model._validate_controls()
    decoder = state['decoder']
    if cls is QuantileSparseRegressor:
        if type(decoder) is not dict or set(decoder) != {'boundaries'} or type(decoder['boundaries']) is not list:
            raise ValueError('Exact quantile decoder required')
        if any(type(v) not in (int, float) for v in decoder['boundaries']):
            raise ValueError('Numeric quantile boundaries required')
        model.percentiles = np.asarray(decoder['boundaries'], dtype=np.float64)
        if len(model.percentiles) != model.n_bins+1:
            raise ValueError('Quantile decoder bin count mismatch')
        quantile_midpoints(model.percentiles)
    elif model.use_target_scaling:
        if (type(decoder) is not dict or set(decoder) != {'mean', 'scale'}
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in decoder.values())
                or decoder['scale'] <= 0):
            raise ValueError('Finite response mean and positive scale required')
        model.target_scaler.mean_ = np.array([decoder['mean']], dtype=np.float64)
        model.target_scaler.scale_ = np.array([decoder['scale']], dtype=np.float64)
        model.target_scaler.n_features_in_ = 1
    elif decoder is not None:
        raise ValueError('Unscaled MSE mode requires no decoder state')
    if type(state['weights']) is not dict:
        raise ValueError('Named numeric weights required')
    model.n_features_in_ = len(feature_columns)
    model._graph = tf.Graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                           device_count={'CPU': 1, 'GPU': 0})
    try:
        with model._graph.as_default():
            from sciona.sparse_neural_regression import SparseMatrix
            model._xs = SparseMatrix()
            model._build_model(model.n_features_in_)
            variables = tf.trainable_variables()
            if set(state['weights']) != {v.name for v in variables}:
                raise ValueError('Exact learned-variable inventory required')
            assignments = []
            for variable in variables:
                raw = np.asarray(state['weights'][variable.name])
                if raw.dtype.kind not in 'fiu' or raw.shape != tuple(variable.shape.as_list()):
                    raise ValueError('Learned variable shape or numeric type differs')
                with np.errstate(over='ignore', invalid='ignore'):
                    values = raw.astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError('Finite float32 learned weights required')
                assignments.append(tf.assign(variable, values))
            model._session = tf.Session(graph=model._graph, config=config)
            model._session.run(assignments)
        model.is_fitted = True
        return model
    except BaseException:
        model.close()
        raise

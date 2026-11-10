"""Serial two-fold OSIC quantile lifecycle over explicit transductive inputs.

Preserves two contiguous folds, batch size 128 and 600 epochs. Seeds are caller
policy, not a claim to reproduce historical random draws. Feature preparation
is joint with the declared query population; validation is not leakage-free.
"""
import json
import math
import numpy as np

from sciona.osic_quantile_features import prepare_features
from sciona.osic_quantile_model import build_model
from sciona.osic_quantile_state import export_prediction_state, restore_prediction_state
from sciona.osic_predictions import quantile_predictions


def predict_in_batches(model, features):
    if not isinstance(features, np.ndarray) or features.ndim != 2 or not len(features):
        raise ValueError('Nonempty prepared query matrix required')
    return np.concatenate([model(features[start:start+128], training=False).numpy()
                           for start in range(0, len(features), 128)], axis=0)


def known_observations(baseline, queries):
    """Caller-supplied observed baselines only; equality follows the source keys."""
    lookup = {(r.entity, r.time):r.target for r in baseline.itertuples()}
    keys = [(r.entity, r.time) for r in queries.itertuples()]
    return (np.array([k in lookup for k in keys], dtype=bool),
            np.array([lookup.get(k, 0.) for k in keys], dtype=np.float64))


def replay_quantile_branch(states, features, columns, feature_coordinate, response_coordinate,
                           observed_mask, observed_values):
    """Replay the two fixed fold models on identically prepared features.

No scaler fitting or training occurs. Feature preparation and the complete
transductive query population remain separate explicit inputs/provenance.
"""
    predictions = replay_prepared_quantile_folds(states, features, columns,
        feature_coordinate, response_coordinate)
    return quantile_predictions(predictions, observed_mask, observed_values)


def replay_prepared_quantile_folds(states, features, columns, feature_coordinate, response_coordinate):
    """Restore two fixed models and return raw predictions shaped (2,M,3).

Features are a nonempty finite float32 matrix aligned to the explicit ordered
columns and saved feature coordinate. Response units follow the saved response
coordinate. No preparation, training, fold reduction or baseline override occurs.
Saved state is inference-only; authenticity is the caller's responsibility.
"""
    if type(states) is not list or len(states) != 2:
        raise ValueError('Both source fold states required')
    if (not isinstance(features, np.ndarray) or features.dtype != np.float32
            or features.ndim != 2 or min(features.shape) < 1
            or type(columns) is not list or features.shape[1] != len(columns)
            or not np.isfinite(features).all()):
        raise ValueError('Finite float32 query matrix aligned to saved feature columns required')
    predictions = []
    for state in states:
        model = restore_prediction_state(state, columns, feature_coordinate, response_coordinate)
        predictions.append(predict_in_batches(model, features))
    return np.stack(predictions)


def train_quantile_branch(training, baseline, queries, *, feature_coordinate, response_coordinate,
                          seeds, allow_transductive, progress=None):
    """Domain adapter: prepare joint populations, fit folds and apply baseline policy."""
    prepared = prepare_features(training, baseline, queries, allow_transductive=allow_transductive)
    columns = [json.dumps(c, separators=(',', ':')) for c in prepared['feature_columns']]
    x = prepared['populations']['training']['features'].astype(np.float32)
    y = prepared['populations']['training']['targets'].astype(np.float32).reshape(-1, 1)
    query_features = prepared['populations']['query']['features'].astype(np.float32)
    fitted = train_prepared_quantile_folds(x, y, query_features, columns,
        feature_coordinate=feature_coordinate, response_coordinate=response_coordinate,
        seeds=seeds, progress=progress)
    mask, values = known_observations(baseline, queries)
    output = quantile_predictions(fitted['fold_predictions'], mask, values)
    return dict(predictions=output, oof_predictions=fitted['oof_predictions'],
        fold_states=fitted['fold_states'], folds=fitted['folds'],
        query_features=query_features, feature_columns=columns, preparation=prepared,
        observed_mask=mask, observed_values=values)


def train_prepared_quantile_folds(x, y, query_features, columns, *, feature_coordinate,
                                  response_coordinate, seeds, progress=None):
    """Fit the fixed source architecture/schedule over explicit numeric matrices.

Inputs are finite float32 arrays: features (N,D), targets (N,1), queries (M,D),
N>=2, M>=1 and D>=1. The caller owns feature preparation, aligned column names,
coordinates, units and observation availability. Two contiguous folds, seeds,
600 epochs and batches of 128 are explicit here; no baseline overrides or
cross-fold prediction reduction occurs. Query-dependent preparation must be
declared upstream and is not made leakage-free by splitting the training rows.
Returns raw fold predictions (2,M,3), out-of-fold predictions (N,3), two private
prediction states and aggregate training records. This fixed quantile model
remains a source-specific training operation, not a general model trainer.
"""
    import tensorflow as tf
    import tf_keras as keras
    if (type(seeds) is not tuple or len(seeds) != 2
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)):
        raise ValueError('Two explicit nonnegative 31-bit fold seeds required')
    if (any(not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.ndim != 2
            for a in (x, y, query_features))
            or len(x) < 2 or x.shape[1] < 1 or y.shape != (len(x), 1)
            or len(query_features) < 1 or query_features.shape[1] != x.shape[1]
            or not all(np.isfinite(v).all() for v in (x, y, query_features))):
        raise ValueError('Aligned nonempty finite float32 feature and target matrices required')
    if (type(columns) is not list or len(columns) != x.shape[1]
            or any(type(c) is not str or not c.strip() for c in columns)
            or len(set(columns)) != len(columns)
            or any(type(c) is not str or not c.strip() for c in (feature_coordinate, response_coordinate))):
        raise ValueError('Explicit unique feature columns and nonempty coordinates required')
    oof = np.empty((len(x), 3), dtype=np.float32)
    states, predictions, records = [], [], []
    # sklearn KFold(n_splits=2, shuffle=False) assigns the extra row to fold 0.
    boundaries = (0, (len(x)+1)//2, len(x))
    for fold, seed in enumerate(seeds):
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        validation = np.arange(boundaries[fold], boundaries[fold+1])
        fit = np.setdiff1d(np.arange(len(x)), validation)
        model = build_model(x.shape[1])
        @tf.function(input_signature=[tf.TensorSpec([None, x.shape[1]], tf.float32), tf.TensorSpec([None, 1], tf.float32)])
        def step(features, targets):
            with tf.GradientTape() as tape:
                loss = model.loss(targets, model(features, training=True))
            model.optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss
        validation_loss = None
        for epoch in range(600):
            order = rng.permutation(fit)
            for start in range(0, len(order), 128):
                batch = order[start:start+128]
                loss = float(step(x[batch], y[batch]).numpy())
                if not math.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
            validation_predictions = predict_in_batches(model, x[validation])
            validation_loss = float(model.loss(y[validation], validation_predictions).numpy())
            if not math.isfinite(validation_loss):
                raise ValueError('Nonfinite held-out loss')
            if progress is not None and (epoch+1) % 200 == 0:
                progress(dict(fold=fold, completed_epochs=epoch+1))
        oof[validation] = validation_predictions
        predictions.append(predict_in_batches(model, query_features))
        expected_updates = 600*math.ceil(len(fit)/128)
        if int(model.optimizer.iterations.numpy()) != expected_updates:
            raise ValueError('Incomplete training schedule')
        states.append(export_prediction_state(model, columns, feature_coordinate, response_coordinate))
        records.append(dict(fold=fold, epochs=600, batch_size=128, training_rows=len(fit),
            validation_rows=len(validation), optimizer_updates=expected_updates,
            validation_checks=600, final_validation_loss=validation_loss))
    return dict(fold_predictions=np.stack(predictions), oof_predictions=oof,
                fold_states=states, folds=records)

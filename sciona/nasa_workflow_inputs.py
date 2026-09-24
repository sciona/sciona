"""Corrected domain joins feeding the reusable numerical CDG.

Inputs are caller-owned runtime DataFrames from the public software interface.
No filesystem loading, dataset discovery or source records are embedded here.
"""
import numpy as np
import pandas as pd
from sciona.nasa_feature_adapters import (estimated_departure_features,
    fit_airline_vocabulary, airline_features, arrival_history_features)
from sciona.tabular_contracts import stable_feature_order, ordered_feature_matrix

ETD_FEATURES = ['minutes_until_departure_from_timepoint', 'minutes_until_departure_from_timestamp',
                'mean_departure_from_timepoint', 'std_departure_from_timepoint']
HISTORY_FEATURES = ['taxitime_to_gate_mean', 'taxitime_to_gate_std']


def _tables(records, airport, vocabulary=None):
    if not isinstance(records, dict) or set(records) != {'queries', 'estimates', 'stands', 'arrivals'}:
        raise ValueError('Explicit queries, estimates, stands and arrivals runtime tables required')
    queries = records['queries']
    if not isinstance(queries, pd.DataFrame) or not {'gufi', 'timestamp', 'airport'} <= set(queries):
        raise ValueError('Query identity, clock and airport fields required')
    if not isinstance(airport, str) or len(airport) != 4 or not (queries.airport == airport).all():
        raise ValueError('One explicit four-character airport slot per adapter call required')
    if queries.duplicated(['gufi', 'timestamp']).any():
        raise ValueError('Unique query identities required')
    vocabulary = fit_airline_vocabulary(queries) if vocabulary is None else vocabulary
    etd = estimated_departure_features(queries, records['estimates'])
    codes = airline_features(queries, vocabulary)
    history = arrival_history_features(queries.timestamp, records['stands'], records['arrivals'], airport[-3:])
    return queries.copy(), etd, codes, history, vocabulary


def _join(queries, etd, codes, history, training):
    merged = queries.merge(etd, on=['gufi', 'timestamp'], how='inner' if training else 'left', validate='one_to_one')
    merged = merged.drop(columns='airport').merge(codes, on='gufi', how='left', validate='many_to_one')
    # Empty history still has typed timestamp keys; pandas otherwise uses object.
    history = history.copy()
    history['timestamp'] = pd.to_datetime(history.timestamp)
    merged = merged.merge(history, on='timestamp', how='left', validate='many_to_one')
    merged['unix_time'] = merged.timestamp.astype('datetime64[ns]').astype(np.int64)//10**9
    if training:
        merged = merged.dropna().reset_index(drop=True)
    return merged


def prepare_training_inputs(training_records, airport):
    """Fit vocabulary and apply source training inner/left joins and dropna.

    Returns the numerical graph inputs, persisted vocabulary, and row identities.
    Group codes use sorted entity identities to preserve sklearn group ordering.
    Empty surviving populations fail explicitly; no rows are synthesized.
    """
    queries, etd, codes, history, vocabulary = _tables(training_records, airport)
    if 'minutes_until_pushback' not in queries:
        raise ValueError('Training target required')
    frame = _join(queries, etd, codes, history, True)
    if frame.empty:
        raise ValueError('No complete training feature rows')
    names = stable_feature_order(['unix_time']+ETD_FEATURES+['Other']+list(vocabulary)+HISTORY_FEATURES)
    features = ordered_feature_matrix(frame, names)
    targets = frame.minutes_until_pushback.to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError('Finite training targets required')
    _, groups = np.unique(frame.gufi.to_numpy(), return_inverse=True)
    return dict(features=features, targets=targets, groups=groups.astype(np.int64), feature_names=names), vocabulary, frame[['gufi', 'timestamp', 'airport']].copy()


def prepare_prediction_inputs(prediction_records, airport, feature_names, vocabulary):
    """Use the training vocabulary and schema; retain query rows with missing features.

    Missing ETD/history values remain NaN for the backend's explicit missing-value
    behavior. Original query order is restored after joins, including empty history.
    """
    queries, etd, codes, history, _ = _tables(prediction_records, airport, vocabulary)
    frame = _join(queries, etd, codes, history, False)
    identities = queries[['gufi', 'timestamp', 'airport']].reset_index(drop=True)
    frame = identities.merge(frame, on=['gufi', 'timestamp', 'airport'], how='left', validate='one_to_one')
    if len(frame) != len(queries) or not frame.columns.is_unique or not set(feature_names) <= set(frame):
        raise ValueError('Prediction row or trained feature coverage differs')
    values = frame[feature_names].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError('Infinite prediction features rejected')
    return values, identities


def attach_predictions(identities, predictions):
    """Attach aligned int32 graph outputs without modifying runtime identity rows."""
    values = np.asarray(predictions)
    if values.dtype != np.int32 or values.shape != (len(identities),):
        raise ValueError('One int32 prediction per query identity required')
    result = identities.copy()
    result['minutes_until_pushback'] = values
    return result

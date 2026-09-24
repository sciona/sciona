"""Explicitly corrected NASA Phase 1 feature adapters; not catalog-approved.

These adapters use the public software's field conventions. They correct final
group omission, absent fallback categories and historical observation leakage.
They are not an exact reproduction of the original complete workflow.
All timestamps must be timezone-naive datetime64 values in one declared clock.
"""
import numpy as np
import pandas as pd

from sciona.atoms.ml.calibration.adaptive_history import adaptive_history_statistics

ETD_COLUMNS = ['gufi', 'timestamp', 'minutes_until_departure_from_timepoint',
    'minutes_until_departure_from_timestamp', 'mean_departure_from_timepoint',
    'median_departure_from_timepoint', 'std_departure_from_timepoint', 'found_etd_vals']


def _check(frame, columns, times=()):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not set(columns) <= set(frame.columns):
        raise ValueError('Unique columns and required adapter fields expected')
    if frame[list(columns)].isna().any().any():
        raise ValueError('Missing adapter values are not supported')
    if 'gufi' in columns and not frame.gufi.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError('Nonempty string entity identifiers required')
    for name in times:
        dtype = frame[name].dtype
        if not pd.api.types.is_datetime64_dtype(dtype):
            raise ValueError('Timezone-naive datetime64 timestamps required')


def estimated_departure_features(queries, estimates):
    """All eligible entities, strict past observation/future estimate, no imputation.

    Last means greatest observation time; equal-time ties preserve input order.
    No eligible estimates produces no feature row, as in the source per-query
    computation. Queries must be unique so downstream joins cannot multiply them.
    """
    _check(queries, ['gufi', 'timestamp'], ['timestamp'])
    _check(estimates, ['gufi', 'timestamp', 'departure_runway_estimated_time'],
           ['timestamp', 'departure_runway_estimated_time'])
    if queries.duplicated(['gufi', 'timestamp']).any():
        raise ValueError('Unique entity/query pairs required')
    groups = {key: group.sort_values('timestamp', kind='stable')
              for key, group in estimates.groupby('gufi', sort=False)}
    rows = []
    for query in queries.itertuples(index=False):
        group = groups.get(query.gufi)
        if group is None:
            continue
        selected = group[(group.timestamp < query.timestamp)
                         & (group.departure_runway_estimated_time > query.timestamp)]
        if selected.empty:
            continue
        remaining = (selected.departure_runway_estimated_time-query.timestamp).dt.total_seconds().to_numpy()/60
        last = selected.iloc[-1]
        rows.append([query.gufi, query.timestamp, float(remaining[-1]),
            (last.departure_runway_estimated_time-last.timestamp).total_seconds()/60,
            float(np.mean(remaining)), float(np.median(remaining)), float(np.std(remaining)), len(selected)])
    return pd.DataFrame(rows, columns=ETD_COLUMNS)


def fit_airline_vocabulary(entities, maximum=25):
    """Select frequent three-character prefixes once, with lexical tie breaking.

    Counts use unique entities, matching source deduplication. Lexical tie order
    is explicit and may differ from pandas' source value_counts tie ordering.
    The returned immutable vocabulary is reused at inference.
    """
    _check(entities, ['gufi'])
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Positive integer vocabulary size required')
    identities = entities.gufi.drop_duplicates()
    if (identities.str.len() < 3).any():
        raise ValueError('Three-character prefixes required')
    counts = identities.str[:3].value_counts()
    return tuple(sorted(counts.index, key=lambda code: (-int(counts[code]), code))[:maximum])


def airline_features(entities, vocabulary):
    """Encode unique entities with a fixed vocabulary and an always-present Other.

    No refitting on inference inputs and no mutation of the supplied vocabulary.
    """
    _check(entities, ['gufi', 'airport'])
    if not isinstance(vocabulary, tuple) or any(not isinstance(v, str) or len(v) != 3 for v in vocabulary) or len(set(vocabulary)) != len(vocabulary):
        raise ValueError('Unique immutable three-character vocabulary required')
    if (entities.gufi.str.len() < 3).any() or (entities.groupby('gufi').airport.nunique() > 1).any():
        raise ValueError('Consistent airport and valid prefix per entity required')
    result = entities[['gufi', 'airport']].drop_duplicates('gufi').reset_index(drop=True).copy()
    codes = result.gufi.str[:3]
    result['Other'] = (~codes.isin(vocabulary)).astype(np.int64)
    for code in vocabulary:
        result[code] = (codes == code).astype(np.int64)
    return result


def arrival_history_features(query_times, stands, arrivals, airport_short):
    """As-of arrival snapshots, followed by the reusable adaptive-history atom.

    Both observations must precede the query. For each entity, select each
    stream's latest available record; equal-time revisions are rejected as
    ambiguous. This explicitly replaces the source many-to-many history join.
    Stand/runway observations are complete snapshots, not sparse updates.
    Only destination-matching, completed, nonnegative durations enter history.
    Zero-count rows are omitted as in the source feature export.
    """
    queries = pd.DataFrame({'timestamp': query_times})
    _check(queries, ['timestamp'], ['timestamp'])
    _check(stands, ['gufi', 'timestamp', 'arrival_stand_actual_time'],
           ['timestamp', 'arrival_stand_actual_time'])
    _check(arrivals, ['gufi', 'timestamp', 'arrival_runway_actual_time'],
           ['timestamp', 'arrival_runway_actual_time'])
    if not isinstance(airport_short, str) or len(airport_short) != 3:
        raise ValueError('Explicit three-character destination required')
    for frame in (stands, arrivals):
        if frame.duplicated(['gufi', 'timestamp']).any():
            raise ValueError('Ambiguous equal-time observation revisions')
    parts = stands.gufi.str.split('.')
    if not parts.map(lambda p: len(p) >= 3 and len(p[2]) == 3).all():
        raise ValueError('Source entity destination syntax required')
    destination = parts.map(lambda p: p[2]) == airport_short
    rows = []
    for query in queries.timestamp.drop_duplicates().sort_values():
        s = stands[destination & (stands.timestamp < query)].sort_values('timestamp', kind='stable').drop_duplicates('gufi', keep='last')
        a = arrivals[arrivals.timestamp < query].sort_values('timestamp', kind='stable').drop_duplicates('gufi', keep='last')
        joined = s[['gufi', 'arrival_stand_actual_time']].merge(
            a[['gufi', 'arrival_runway_actual_time']], on='gufi', validate='one_to_one')
        joined = joined[joined.arrival_stand_actual_time < query]
        durations = (joined.arrival_stand_actual_time-joined.arrival_runway_actual_time).dt.total_seconds().to_numpy(dtype=np.float64)/60
        if (durations < 0).any():
            raise ValueError('Negative completed arrival duration')
        times = joined.arrival_stand_actual_time.astype('datetime64[ns]').astype(np.int64).to_numpy()
        count, mean, std = adaptive_history_statistics(times, durations, int(query.value),
            np.array([60, 120, 180], dtype=np.int64)*60*10**9, 3)
        if count:
            rows.append([query, count, mean, std])
    return pd.DataFrame(rows, columns=['timestamp', 'found_counts_taxitime_to_gate',
                                     'taxitime_to_gate_mean', 'taxitime_to_gate_std'])

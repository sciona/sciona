"""Explicit domain partitioning and identity-safe reassembly for runtime tables.

These adapters do not fit models or choose numerical policies. Each caller-owned
airport slot executes its own decomposed feature and numerical workflow.
"""
import numpy as np
import pandas as pd

from sciona.nasa_feature_adapters import _check

IDENTITY = ['gufi', 'timestamp', 'airport']
RECORD_KEYS = {'queries', 'estimates', 'stands', 'arrivals'}


def partition_records(records, airports):
    """Partition query rows, keeping event streams available to each adapter.

    Arrival histories can involve entities absent from the prediction queries;
    filtering event streams by query identity would silently remove history.
    The downstream feature adapters enforce event/observation cutoffs and
    destination selection. Event tables are borrowed read-only runtime inputs.
    """
    if not isinstance(records, dict) or set(records) != RECORD_KEYS:
        raise ValueError('Explicit query and event tables required')
    if not isinstance(airports, tuple) or not airports or any(
            not isinstance(slot, str) or len(slot) != 4 for slot in airports) or len(set(airports)) != len(airports):
        raise ValueError('Unique explicit four-character airport slots required')
    queries = records['queries']
    _check(queries, IDENTITY, ['timestamp'])
    if queries.empty or queries.duplicated(IDENTITY).any():
        raise ValueError('Nonempty unique query identities required')
    if not queries.airport.isin(airports).all():
        raise ValueError('Query airport is absent from declared slots')
    if (queries.groupby('gufi').airport.nunique() > 1).any():
        raise ValueError('One airport per entity required')
    for name in RECORD_KEYS - {'queries'}:
        if not isinstance(records[name], pd.DataFrame):
            raise ValueError('Runtime event DataFrames required')
    partitions = {}
    for airport in airports:
        selected = queries.loc[queries.airport == airport].copy().reset_index(drop=True)
        if selected.empty:
            continue
        partitions[airport] = dict(records, queries=selected)
    return partitions, queries[IDENTITY].copy().reset_index(drop=True)


def assemble_predictions(identities, slot_predictions):
    """Require exactly one int32 result per input identity and restore order.

    No slot order, DataFrame index or concatenation order is used as identity.
    There is no implicit default for a missing population or failed model.
    """
    _check(identities, IDENTITY, ['timestamp'])
    if identities.empty or identities.duplicated(IDENTITY).any():
        raise ValueError('Nonempty unique output identities required')
    slots = set(identities.airport)
    if not isinstance(slot_predictions, dict) or set(slot_predictions) != slots:
        raise ValueError('Exactly the queried airport results are required')
    pieces = []
    for airport, frame in slot_predictions.items():
        _check(frame, IDENTITY, ['timestamp'])
        if set(frame) != set(IDENTITY + ['minutes_until_pushback']) or not (frame.airport == airport).all():
            raise ValueError('Result fields or airport identity differ')
        if frame.minutes_until_pushback.dtype != np.dtype('int32'):
            raise ValueError('Int32 model outputs required')
        pieces.append(frame)
    combined = pd.concat(pieces, ignore_index=True)
    if len(combined) != len(identities) or combined.duplicated(IDENTITY).any():
        raise ValueError('Exactly one prediction per query required')
    coverage = identities[IDENTITY].merge(combined[IDENTITY], on=IDENTITY, how='outer',
                                          validate='one_to_one', indicator=True)
    if not coverage['_merge'].eq('both').all():
        raise ValueError('Prediction identity coverage differs')
    result = identities[IDENTITY].merge(combined, on=IDENTITY, how='left', sort=False, validate='one_to_one')
    pd.testing.assert_frame_equal(result[IDENTITY].reset_index(drop=True), identities[IDENTITY].reset_index(drop=True))
    return result

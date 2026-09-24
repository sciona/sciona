"""Structured identifier features with explicit format and category policies."""
import re

import numpy as np
import pandas as pd


def structured_identity_features(query_ids, query_times, record_ids, *, delimiter,
        minimum_parts, primary_part, category_part, reference_parts,
        reference_format, vocabularies, fallback, seconds_per_unit):
    """Return lexical features and signed time from an encoded reference.

    Every identifier has at least minimum_parts nonempty delimiter-separated
    fields. Reference fields concatenate in the declared order and parse with
    reference_format. Times are naive, on the same caller-declared clock.
    Only identifiers in the authoritative unique record snapshot receive time
    features; unknown records retain NaN. Snapshot availability is a caller
    obligation. Vocabulary keys are primary, numeric and category. Numeric
    extraction follows primary fallback, and numeric length/final character
    follow numeric fallback, preserving that explicitly declared source policy.
    Alphabetic prefix removes ASCII digits from the original primary token.
    """
    for ids in (query_ids, record_ids):
        if (not isinstance(ids, np.ndarray) or ids.ndim != 1
                or not all(isinstance(value, str) and value for value in ids)):
            raise ValueError('Nonempty string identifiers required')
    if len(set(record_ids)) != len(record_ids):
        raise ValueError('Unique authoritative record identifiers required')
    if (not isinstance(query_times, pd.DatetimeIndex) or query_times.tz is not None
            or query_times.hasnans or len(query_times) != len(query_ids)):
        raise ValueError('Aligned nonmissing naive query times required')
    if not isinstance(delimiter, str) or not delimiter or type(minimum_parts) is not int or minimum_parts < 2:
        raise ValueError('Explicit delimiter and minimum field count required')
    if not isinstance(reference_parts, list) or not reference_parts:
        raise ValueError('Ordered reference field indices required')
    if any(type(index) is not int or not 0 <= index < minimum_parts for index in [primary_part, category_part, *reference_parts]):
        raise ValueError('Field indices must be within the minimum field count')
    if not isinstance(reference_format, str) or not reference_format or not isinstance(fallback, str) or not fallback:
        raise ValueError('Explicit reference format and nonempty fallback required')
    if type(seconds_per_unit) not in (float, int) or not np.isfinite(seconds_per_unit) or seconds_per_unit <= 0:
        raise ValueError('Finite positive time unit required')
    if not isinstance(vocabularies, dict) or set(vocabularies) != {'primary', 'numeric', 'category'}:
        raise ValueError('Explicit primary, numeric and category vocabularies required')
    for vocabulary in vocabularies.values():
        if (not isinstance(vocabulary, list) or not all(isinstance(v, str) for v in vocabulary)
                or len(set(vocabulary)) != len(vocabulary)):
            raise ValueError('Unique string vocabulary lists required')
    parts = {}
    for identity in [*record_ids, *query_ids]:
        fields = identity.split(delimiter)
        if len(fields) < minimum_parts or not all(fields):
            raise ValueError('Identifier does not satisfy the declared field format')
        parts[identity] = fields
    identities = list(parts)
    references = pd.to_datetime([''.join(parts[key][i] for i in reference_parts) for key in identities],
        format=reference_format, exact=True, errors='raise')
    if references.tz is not None or references.hasnans:
        raise ValueError('Nonmissing naive encoded reference times required')
    reference_by_id = dict(zip(identities, references))
    authoritative = set(record_ids)
    rows = []
    for identity, query in zip(query_ids, query_times):
        fields = parts[identity]
        raw = fields[primary_part]
        primary = raw if raw in vocabularies['primary'] else fallback
        numeric = ''.join(filter(str.isdigit, primary))
        numeric = numeric if numeric in vocabularies['numeric'] else fallback
        category = fields[category_part]
        reference = reference_by_id[identity] if identity in authoritative else None
        elapsed = np.nan if reference is None else (int(query.value)-int(reference.value))/1e9/seconds_per_unit
        if reference is not None and not np.isfinite(elapsed):
            raise ValueError('Elapsed time outside supported finite arithmetic range')
        rows.append([primary, re.sub(r'[0-9]+', '', raw), numeric, len(numeric),
            numeric[-1] if numeric else fallback,
            category if category in vocabularies['category'] else fallback,
            fields[-1], fields[-2], np.nan if reference is None else reference.hour, elapsed])
    return pd.DataFrame(rows, columns=['primary', 'prefix', 'numeric', 'numeric_length',
        'numeric_last', 'category', 'last', 'penultimate', 'reference_hour', 'elapsed'])

"""Explicit vocabulary projection from authoritative unique entity snapshots."""
import numpy as np
import pandas as pd


def unique_entity_features(query_ids, record_ids, attributes, vocabularies, fallback):
    """Project snapshot attributes onto query identities without changing rows.

    attributes is a DataFrame aligned positionally with record_ids. vocabularies
    maps categorical attribute names to caller-reviewed string lists; fields not
    listed pass through unchanged. Unknown identities yield fallback categorical
    values and missing passthrough values. Repeated queries and their order are
    preserved. Duplicate source identities are rejected, even for equal records.
    The caller must supply a snapshot authoritative at the query cutoff; this
    operation does not infer availability or resolve temporal record updates.
    Returns a DataFrame with a fresh positional index, without mutating inputs.
    """
    for ids in (query_ids, record_ids):
        if (not isinstance(ids, np.ndarray) or ids.ndim != 1
                or not all(isinstance(v, str) and v for v in ids)):
            raise ValueError('Nonempty string entity identifiers required')
    if len(set(record_ids)) != len(record_ids):
        raise ValueError('Unique authoritative records required')
    if (not isinstance(attributes, pd.DataFrame) or len(attributes) != len(record_ids)
            or not attributes.columns.is_unique):
        raise ValueError('Aligned attributes with unique columns required')
    if not isinstance(fallback, str) or not fallback:
        raise ValueError('Explicit nonempty fallback string required')
    if not isinstance(vocabularies, dict) or any(c not in attributes for c in vocabularies):
        raise ValueError('Vocabulary keys must name attribute columns')
    for vocabulary in vocabularies.values():
        if (not isinstance(vocabulary, list) or not all(isinstance(v, str) for v in vocabulary)
                or len(set(vocabulary)) != len(vocabulary)):
            raise ValueError('Explicit unique string vocabularies required')
    indexed = attributes.copy(deep=True)
    indexed.index = pd.Index(record_ids)
    result = indexed.reindex(query_ids).reset_index(drop=True)
    for column, vocabulary in vocabularies.items():
        values = result[column].astype(object)
        result[column] = values.where(values.isin(vocabulary), fallback)
    return result

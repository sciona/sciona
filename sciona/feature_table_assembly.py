"""Query-preserving, unambiguous feature-table assembly."""
import numpy as np
import pandas as pd


def assemble_feature_tables(queries, feature_tables):
    """Left-join (DataFrame, key-list) pairs without row loss or fanout.

    Query order, duplicate queries, original columns and index are preserved.
    Feature keys must be unique and nonmissing; overlapping non-key columns are
    rejected. Missing matches remain missing for a separately declared imputation
    policy. No forward/backward fill or inferred aggregation is performed.
    """
    if not isinstance(queries, pd.DataFrame) or not queries.columns.is_unique:
        raise ValueError('Query DataFrame with unique columns required')
    if not isinstance(feature_tables, list):
        raise ValueError('Explicit list of feature tables and keys required')
    result = queries.copy(deep=True)
    marker = object()
    result[marker] = np.arange(len(result))
    for specification in feature_tables:
        if not isinstance(specification, tuple) or len(specification) != 2:
            raise ValueError('Each feature specification must be (table, keys)')
        table, keys = specification
        if not isinstance(table, pd.DataFrame) or not table.columns.is_unique:
            raise ValueError('Feature DataFrame with unique columns required')
        if (not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys)
                or len(set(keys)) != len(keys) or any(k not in queries or k not in table for k in keys)):
            raise ValueError('Unique explicit keys present in queries and feature table required')
        if queries[keys].isna().any().any() or table[keys].isna().any().any():
            raise ValueError('Missing join keys are not permitted')
        if table.duplicated(keys).any():
            raise ValueError('Feature keys must be unique to prevent query fanout')
        if (set(table.columns) - set(keys)) & set(result.columns):
            raise ValueError('Overlapping feature columns require explicit renaming')
        result = result.merge(table, on=keys, how='left', sort=False, validate='many_to_one')
    result = result.sort_values(marker, kind='stable').drop(columns=[marker])
    if len(result) != len(queries):
        raise ValueError('Feature assembly changed query count')
    result.index = queries.index.copy()
    return result

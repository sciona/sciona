"""Explicit source feature recipe using neutral, caller-mapped table fields.

No source records, identifiers or file paths are embedded. The caller maps
domain fields to entity/time/target/numeric/category_a/category_b. Query rows
contain entity/time only; metadata and the observed response come from baseline.
This recipe uses query-population information and is explicitly transductive.
"""
import numpy as np
import pandas as pd

FIELDS = ['entity', 'time', 'target', 'numeric', 'category_a', 'category_b']


def _population(frame, columns):
    if not isinstance(frame, pd.DataFrame) or frame.empty or set(frame.columns) != set(columns):
        raise ValueError('Nonempty explicitly mapped table with exact fields required')
    if frame.isna().any().any():
        raise ValueError('Complete rows required')
    for field in ('entity', 'category_a', 'category_b'):
        if field in columns and any(type(x) is not str or not x.strip() for x in frame[field]):
            raise ValueError('Nonblank string entity and category labels required')
    for field in ('time', 'target', 'numeric'):
        if field in columns:
            values = frame[field].to_numpy()
            if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
                raise ValueError('Finite real numeric coordinates required')


def assemble_populations(training, baseline, queries):
    """Remove all repeated training entity/time rows and join query metadata.

Exactly one supplied baseline per queried entity is required. Training and
baseline populations may overlap; source encounter order remains significant.
No query key parsing, availability inference or clinical feature extraction.
"""
    _population(training, FIELDS)
    _population(baseline, FIELDS)
    _population(queries, ['entity', 'time'])
    if baseline['entity'].duplicated().any() or not set(queries['entity']) <= set(baseline['entity']):
        raise ValueError('Unique supplied baseline for every query entity required')
    train = training.drop_duplicates(subset=['entity', 'time'], keep=False).copy()
    if train.empty:
        raise ValueError('Deduplication removed the complete training population')
    base = baseline.copy()
    query = queries.merge(base.drop(columns='time'), on='entity', how='left', sort=False, validate='many_to_one')
    train['population'] = 'training'
    base['population'] = 'baseline'
    query['population'] = 'query'
    return pd.concat([train, base, query], ignore_index=True)


def joint_population_features(population, *, allow_transductive):
    """Fit source baseline/category/min-max features over all supplied rows.

Returns matrices by population, their entity/time order, and explicit fitted
feature metadata. Categories follow first encounter order, separately per
categorical field. Numeric output order is numeric, time offset, baseline target.
Only nondegenerate, finite scaling ranges are supported. This is not a reusable
train-only scaler: future query changes can change training feature values.
"""
    if allow_transductive is not True:
        raise ValueError('Explicit authorization to use the query population is required')
    _population(population, FIELDS+['population'])
    if set(population['population']) != {'training', 'baseline', 'query'}:
        raise ValueError('All three explicit population roles required')
    data = population.copy()
    data['baseline_time'] = data['time'].where(data['population'] != 'query')
    data['baseline_time'] = data.groupby('entity')['baseline_time'].transform('min')
    earliest = data.loc[data['time'] == data['baseline_time'], ['entity', 'target']]
    earliest = earliest.drop_duplicates('entity', keep='first').rename(columns={'target': 'baseline_target'})
    data = data.merge(earliest, on='entity', how='left', sort=False, validate='many_to_one')
    data['time_offset'] = data['time'] - data['baseline_time']
    if not np.isfinite(data[['baseline_target', 'time_offset']].to_numpy()).all():
        raise ValueError('Observed baseline required for every entity')
    features = []
    columns = []
    categories = {}
    for field in ('category_a', 'category_b'):
        categories[field] = data[field].unique().tolist()
        for label in categories[field]:
            features.append((data[field] == label).to_numpy(dtype=np.int64))
            columns.append([field, label])
    # The original notebook writes categories directly as table column names.
    # Explicitly reject cross-field collisions rather than silently correcting them.
    if set(categories['category_a']) & set(categories['category_b']):
        raise ValueError('Source category-name collision requires separate review')
    ranges = {}
    for field in ('numeric', 'time_offset', 'baseline_target'):
        values = data[field].to_numpy()
        lower, upper = values.min(), values.max()
        if not upper > lower:
            raise ValueError('Source min-max scaling requires a nonzero range')
        normalized = (values-lower)/(upper-lower)
        if not np.isfinite(normalized).all():
            raise ValueError('Finite scaled features required')
        features.append(normalized)
        columns.append([field])
        ranges[field] = [float(lower), float(upper)]
    matrix = np.column_stack(features)
    result = {}
    for role in ('training', 'baseline', 'query'):
        mask = (data['population'] == role).to_numpy()
        result[role] = dict(features=matrix[mask].copy(), targets=data.loc[mask, 'target'].to_numpy().copy(),
            entities=data.loc[mask, 'entity'].tolist(), times=data.loc[mask, 'time'].to_numpy().copy())
    return dict(populations=result, feature_columns=columns, categories=categories, ranges=ranges,
                transductive=True)


def prepare_features(training, baseline, queries, *, allow_transductive):
    return joint_population_features(assemble_populations(training, baseline, queries),
                                     allow_transductive=allow_transductive)

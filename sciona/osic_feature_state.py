"""Private numeric preprocessing state for a declared transductive query batch.

State contains caller data and must stay outside Git. It retains category order,
scaling ranges, observed baselines and query scope, but no training records or
precomputed feature matrix. Restoration performs no fitting or extrema discovery.
"""
import copy
import numpy as np
from sciona.osic_quantile_features import assemble_populations, joint_population_features, _population

FORMAT = 'sciona.osic_joint_feature_state.v1'
NUMERIC = ('numeric', 'time_offset', 'baseline_target')
DTYPES = {'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'}


def _coordinates(feature_coordinate, response_coordinate):
    if any(type(v) is not str or not v.strip() for v in (feature_coordinate, response_coordinate)):
        raise ValueError('Explicit feature and response coordinates required')


def export_preparation_state(training, baseline, queries, *, feature_coordinate,
                             response_coordinate, allow_transductive):
    _coordinates(feature_coordinate, response_coordinate)
    data = assemble_populations(training, baseline, queries)
    prepared = joint_population_features(data, allow_transductive=allow_transductive)
    data['baseline_time'] = data['time'].where(data['population'] != 'query')
    data['baseline_time'] = data.groupby('entity')['baseline_time'].transform('min')
    earliest = data.loc[data['time'] == data['baseline_time']].drop_duplicates('entity', keep='first').set_index('entity')
    offset = data['time']-data['baseline_time']
    observations = []
    for row in baseline.to_dict('records'):
        if row['entity'] not in set(queries['entity']):
            continue
        initial = earliest.loc[row['entity']]
        observations.append(dict(entity=row['entity'], numeric=row['numeric'],
            category_a=row['category_a'], category_b=row['category_b'],
            observed_time=row['time'], observed_target=row['target'],
            baseline_time=initial['baseline_time'].item(), baseline_target=initial['target'].item()))
    state = dict(format=FORMAT, feature_coordinate=feature_coordinate, response_coordinate=response_coordinate,
        categories=copy.deepcopy(prepared['categories']), ranges=copy.deepcopy(prepared['ranges']),
        dtypes=dict(numeric=str(data['numeric'].dtype), time_offset=str(offset.dtype),
                    baseline_target=str(data['target'].dtype)),
        feature_columns=copy.deepcopy(prepared['feature_columns']), observations=observations,
        query_scope=queries.to_dict('records'))
    restored = transform_queries(state, queries, feature_coordinate=feature_coordinate, response_coordinate=response_coordinate)
    np.testing.assert_array_equal(restored['features'], prepared['populations']['query']['features'])
    return state


def transform_queries(state, queries, *, feature_coordinate, response_coordinate):
    _coordinates(feature_coordinate, response_coordinate)
    _population(queries, ['entity', 'time'])
    fields = {'format','feature_coordinate','response_coordinate','categories','ranges',
              'dtypes','feature_columns','observations','query_scope'}
    if (type(state) is not dict or set(state) != fields or state['format'] != FORMAT
            or state['feature_coordinate'] != feature_coordinate or state['response_coordinate'] != response_coordinate
            or state['query_scope'] != queries.to_dict('records')):
        raise ValueError('Exact format, coordinates and declared ordered query batch required')
    categories = state['categories']
    if type(categories) is not dict or set(categories) != {'category_a', 'category_b'}:
        raise ValueError('Exact categorical fields required')
    for values in categories.values():
        if (type(values) is not list or not values or any(type(v) is not str or not v.strip() for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Nonempty unique ordered category labels required')
    if set(categories['category_a']) & set(categories['category_b']):
        raise ValueError('Cross-field category collision')
    columns = [[field, value] for field in ('category_a', 'category_b') for value in categories[field]]
    columns += [[field] for field in NUMERIC]
    if state['feature_columns'] != columns:
        raise ValueError('Feature order differs from fitted categories')
    if (type(state['dtypes']) is not dict or set(state['dtypes']) != set(NUMERIC)
            or any(value not in DTYPES for value in state['dtypes'].values())
            or type(state['ranges']) is not dict or set(state['ranges']) != set(NUMERIC)):
        raise ValueError('Known numeric dtypes and complete fitted ranges required')
    for bounds in state['ranges'].values():
        if (type(bounds) is not list or len(bounds) != 2
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in bounds)
                or not bounds[1] > bounds[0]):
            raise ValueError('Finite nondegenerate fitted ranges required')
    observation_fields = {'entity','numeric','category_a','category_b','observed_time','observed_target','baseline_time','baseline_target'}
    if type(state['observations']) is not list or not state['observations']:
        raise ValueError('Stored observations required')
    lookup = {}
    for row in state['observations']:
        if (type(row) is not dict or set(row) != observation_fields
                or type(row['entity']) is not str or row['entity'] in lookup):
            raise ValueError('Unique exact stored observation records required')
        for field in ('category_a', 'category_b'):
            if row[field] not in categories[field]:
                raise ValueError('Stored category is outside fitted vocabulary')
        for field in ('numeric','observed_time','observed_target','baseline_time','baseline_target'):
            if type(row[field]) not in (int, float) or not np.isfinite(row[field]):
                raise ValueError('Finite numeric stored coordinates required')
        lookup[row['entity']] = row
    if set(lookup) != set(queries['entity']):
        raise ValueError('Exact baseline coverage for declared queries required')
    selected = [lookup[e] for e in queries['entity']]
    features = []
    for field in ('category_a', 'category_b'):
        for label in categories[field]:
            features.append(np.array([row[field] == label for row in selected], dtype=np.int64))
    # Keep the original numeric arithmetic dtype before column stacking.
    raw = dict(numeric=[r['numeric'] for r in selected], baseline_target=[r['baseline_target'] for r in selected])
    time_dtype = state['dtypes']['time_offset']
    raw['time_offset'] = queries['time'].to_numpy().astype(time_dtype) - np.array([r['baseline_time'] for r in selected], dtype=time_dtype)
    for field in NUMERIC:
        dtype = state['dtypes'][field]
        values = np.asarray(raw[field], dtype=dtype)
        low, high = np.asarray(state['ranges'][field], dtype=dtype)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            scaled = (values-low)/(high-low)
        if not np.isfinite(scaled).all():
            raise ValueError('Finite restored features required')
        features.append(scaled)
    mask = np.array([time == row['observed_time'] for time,row in zip(queries['time'], selected)], dtype=bool)
    values = np.array([row['observed_target'] if match else 0. for row,match in zip(selected, mask)])
    return dict(features=np.column_stack(features), feature_columns=copy.deepcopy(columns),
                observed_mask=mask, observed_values=values)

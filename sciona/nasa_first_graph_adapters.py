"""Domain field preparation and output projection for decomposed feature graphs.

Extracted from the qualified adapter; generic computations execute in separate
registered atoms. Runtime policies and records remain caller-owned.
"""
import numpy as np
import pandas as pd
from sciona.nasa_first_feature_adapter import _times, MINUTE, WEATHER_FIELDS

OPERATIONS = [('lexical', 'identifier_features'), ('attributes', 'entity_attributes'), ('configuration', 'configuration_features'), ('estimates', 'estimate_features'), ('calendar', 'calendar_features'), ('forecasts', 'forecast_features'), ('counts', 'event_counts'), ('delay', 'event_delays')]

def prepare_arguments(queries, raw_options):
    mfs = raw_options['mfs']
    config = raw_options['config']
    etd = raw_options['etd']
    lamp = raw_options['lamp']
    runways = raw_options['runways']
    standtimes = raw_options['standtimes']
    start_time = raw_options['start_time']
    end_time = raw_options['end_time']
    vocabularies = raw_options['vocabularies']
    holiday_midnights = raw_options['holiday_midnights']
    entity_snapshot_available_at = raw_options['entity_snapshot_available_at']
    _times(queries['timestamp'])
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    snapshot = pd.Timestamp(entity_snapshot_available_at)
    if any((pd.isna(v) or v.tz is not None for v in (start, end, snapshot))) or start > end:
        raise ValueError('Valid naive context and snapshot times required')
    if len(queries) and (queries['timestamp'].min() < start or queries['timestamp'].max() > end or snapshot > queries['timestamp'].min()):
        raise ValueError('Queries must lie in context with an available entity snapshot')
    pairs = queries[['gufi', 'timestamp']].drop_duplicates().reset_index(drop=True)
    times = pairs[['timestamp']].drop_duplicates().reset_index(drop=True)
    query_ticks = _times(times['timestamp'])
    lexical_arguments = (pairs['gufi'].to_numpy(), pd.DatetimeIndex(pairs['timestamp']), mfs['gufi'].to_numpy(), '.', 6, 0, 2, [3, 4], '%y%m%d%H%M', {'primary': vocabularies['FLIGHT_IDS'], 'numeric': vocabularies['FLIGHT_NUMBERS'], 'category': vocabularies['AIRPORTS']}, 'OTHER', 60,)
    attribute_map = {'aircraft_engine_class': ('mfs_cat_engineclass', 'ENGINE_CLASSES'), 'aircraft_type': ('mfs_cat_type', 'AIRCRAFT_TYPES'), 'major_carrier': ('mfs_cat_majorcarrier', 'CARRIER_TYPES'), 'flight_type': ('mfs_cat_flighttype', 'FLIGHT_TYPES'), 'isdeparture': ('mfs_departure', None)}
    attributes_arguments = (pairs['gufi'].to_numpy(), mfs['gufi'].to_numpy(), mfs[list(attribute_map)], {key: vocabularies[rule] for key, (_, rule) in attribute_map.items() if rule}, 'OTHER',)
    grid = pd.date_range(start, end, freq='5min')
    config_windows = np.array([6, 12, 24, 36, 48, 120], dtype=np.int64)
    configuration_arguments = (_times(config['timestamp']), config[['departure_runways', 'arrival_runways']].to_numpy(), _times(grid), vocabularies['RUNWAYS'], ['R', 'L', 'C'], config_windows, 5 * MINUTE,)
    windows = np.array([60, 120, 240, 360, 720, 1440], dtype=np.int64)
    estimates_arguments = (etd['gufi'].to_numpy(), _times(etd['timestamp']), _times(etd['departure_runway_estimated_time'], missing=True), etd['departure_runway_estimated_time'].notna().to_numpy(), pairs['gufi'].to_numpy(), _times(pairs['timestamp']), windows * MINUTE, 15 * MINUTE, MINUTE,)
    calendar_arguments = (pd.DatetimeIndex(times['timestamp']), holiday_midnights,)
    forecast_values = lamp[WEATHER_FIELDS].copy()
    forecast_values['lightning_prob'] = forecast_values['lightning_prob'].map({'L': 0, 'M': 1, 'N': 2, 'H': 3})
    forecast_values['cloud'] = forecast_values['cloud'].map({'OV': 4, 'BK': 3, 'CL': 0, 'FW': 1, 'SC': 2}).fillna(3)
    forecast_values['precip'] = forecast_values['precip'].astype(float)
    horizons = [1, 3, 6, 9, 12, 24]
    forecasts_arguments = (_times(lamp['timestamp']), _times(lamp['forecast_timestamp']), forecast_values.to_numpy(dtype=np.float64), query_ticks, 360 * MINUTE, [(None, 210 * MINUTE), (210 * MINUTE, 390 * MINUTE)], np.array([[30, 60 * h + 30] for h in horizons], dtype=np.int64) * MINUTE, np.arange(1, 7, dtype=np.int64),)
    event_windows = np.array([15, 30, 60, 90, 120, 360, 720, 1440], dtype=np.int64)
    actual_columns = ['departure_runway_actual_time', 'arrival_runway_actual_time']
    countable = np.column_stack([runways['gufi'].notna() & runways[category].notna() & runways[actual].notna() for category, actual in zip(['departure_runway_actual', 'arrival_runway_actual'], actual_columns)])
    counts_arguments = (_times(runways['timestamp']), np.column_stack([_times(runways[c], missing=True) for c in actual_columns]), countable, query_ticks, event_windows * MINUTE,)
    events = standtimes.loc[standtimes['gufi'].notna() & standtimes['departure_stand_actual_time'].notna()]
    delay_arguments = (events['gufi'].to_numpy(), _times(events['timestamp']), _times(events['departure_stand_actual_time']), etd['gufi'].to_numpy(), _times(etd['timestamp']), _times(etd['departure_runway_estimated_time'], missing=True), etd['departure_runway_estimated_time'].notna().to_numpy(), query_ticks, windows * MINUTE, 15 * MINUTE, MINUTE,)
    context = {'queries': queries, 'pairs': pairs, 'times': times, 'grid': grid, 'config_windows': config_windows, 'windows': windows, 'vocabularies': vocabularies, 'attribute_map': attribute_map, 'horizons': horizons, 'event_windows': event_windows}
    return (context, *lexical_arguments, *attributes_arguments, *configuration_arguments, *estimates_arguments, *calendar_arguments, *forecasts_arguments, *counts_arguments, *delay_arguments)

def project_tables(context, lexical, attributes, configuration, estimates, calendar, forecasts, counts, delay_count, delay_value_count, delay_mean, delay_maximum):
    lexical = lexical.copy(deep=True)
    attributes = attributes.copy(deep=True)
    delay = (delay_count, delay_value_count, delay_mean, delay_maximum)
    queries = context['queries']
    pairs = context['pairs']
    times = context['times']
    grid = context['grid']
    config_windows = context['config_windows']
    windows = context['windows']
    vocabularies = context['vocabularies']
    attribute_map = context['attribute_map']
    horizons = context['horizons']
    event_windows = context['event_windows']
    lexical.columns = ['mfs_cat_flightid', 'mfs_cat_airline', 'mfs_cat_flightno', 'mfs_flightno_length', 'mfs_cat_no_last', 'mfs_cat_arrival', 'mfs_cat_last', 'mfs_cat_sectolast', 'mfs_cat_hourdep', 'mfs_time_diff']
    attributes.columns = [value[0] for value in attribute_map.values()]
    entity = pd.concat([pairs, lexical, attributes], axis=1)
    columns = {'timestamp': grid}
    for channel, (long_name, short_name) in enumerate([('departure', 'dep'), ('arrival', 'arr')]):
        for vi, token in enumerate(vocabularies['RUNWAYS']):
            columns[f'config_{long_name}_cat_{token}'] = configuration['categories'][:, channel, vi]
        columns[f'config_n_active_{short_name}'] = configuration['item_counts'][:, channel]
        columns[f'config_change_{short_name}'] = configuration['changes'][:, channel]
        for wi, width in enumerate(config_windows):
            columns[f'config_n_chang_{short_name}_last_{width}'] = configuration['rolling_changes'][:, channel, wi]
    configuration_table = pd.DataFrame(columns).astype({key: float for key in columns if key != 'timestamp'})
    configuration_table.loc[~configuration['state_valid'], configuration_table.columns != 'timestamp'] = np.nan
    estimate_columns = {**{key: pairs[key] for key in pairs}, 'etd_est_change': estimates['change'], 'etd_diff_withfirst': estimates['change_from_first'], 'etd_time_till_est_dep': estimates['remaining']}
    for wi, width in enumerate(windows):
        for stat in ['sum', 'max', 'std']:
            estimate_columns[f'etd_roll_{width}_{stat}'] = estimates[stat][:, wi]
    calendar_names = ['mom_cat_hourofday', 'mom_cat_dayofweek', 'mom_cat_month', 'mom_dayofmonth', 'mom_cat_woy', 'mom_cat_wom', 'mom_cat_isholiday', 'mom_days_until_holiday']
    calendar_table = pd.DataFrame({'timestamp': times['timestamp'], **dict(zip(calendar_names, calendar.values()))})
    weather_columns = {'timestamp': times['timestamp']}
    for fi, name in enumerate(WEATHER_FIELDS):
        for stat in ['mean', 'min', 'max']:
            weather_columns[f'temp_{name}_{stat}_last6h'] = forecasts[stat][:, fi]
        for bi, suffix in enumerate(['next3', 'next36']):
            weather_columns[f'temp_{name}_{suffix}'] = forecasts['bands'][:, bi, fi]
        weather_columns[f'temp_{name}'] = forecasts['latest_valid'][:, fi]
        for lag in range(1, 7):
            weather_columns[f'temp_change_{name}_{lag}'] = forecasts['revisions'][:, lag - 1, fi]
        for hi, horizon in enumerate(horizons):
            weather_columns[f'temp_futchange_{name}_{horizon}'] = forecasts['contrasts'][:, hi, fi]
    count_columns = {'timestamp': times['timestamp']}
    for wi, width in enumerate(event_windows):
        for ci, suffix in enumerate(['totdep', 'totarr']):
            count_columns[f'runways_{width}_{suffix}'] = counts[:, ci, wi]
    delay_columns = {'timestamp': times['timestamp']}
    for wi, hours in enumerate([1, 2, 4, 6, 12, 24]):
        for name, position in [('count', 0), ('meandelay', 2), ('maxdelay', 3)]:
            delay_columns[f'standtimes_{name}_{hours}'] = delay[position][:, wi]
    return (queries, [(entity, ['gufi', 'timestamp']), (configuration_table, ['timestamp']), (pd.DataFrame(estimate_columns), ['gufi', 'timestamp']), (calendar_table, ['timestamp']), (pd.DataFrame(weather_columns), ['timestamp']), (pd.DataFrame(count_columns), ['timestamp']), (pd.DataFrame(delay_columns), ['timestamp'])])

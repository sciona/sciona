"""Corrected public-software field adapter for the first-place feature workflow.

Only software field names and algorithm constants live here. Vocabularies,
calendar schedules, snapshot availability and imputation are runtime policies.
"""
import numpy as np
import pandas as pd

from sciona.calendar_time_features import calendar_time_features
from sciona.configuration_state_features import configuration_state_features
from sciona.estimate_history_features import estimate_history_features
from sciona.event_count_windows import event_count_windows
from sciona.event_delay_windows import event_delay_windows
from sciona.feature_table_assembly import assemble_feature_tables
from sciona.forecast_history_features import forecast_history_features
from sciona.structured_identity_features import structured_identity_features
from sciona.unique_entity_features import unique_entity_features

MINUTE=60_000_000_000
WEATHER_FIELDS=['temperature','wind_direction','wind_speed','wind_gust','cloud_ceiling',
    'visibility','cloud','lightning_prob','precip']


def _times(values, *, missing=False):
    times=pd.DatetimeIndex(values)
    if times.tz is not None or (times.hasnans and not missing):
        raise ValueError('Nonmissing naive timestamps on one declared clock required')
    return times.as_unit('ns').asi8


def nasa_first_feature_tables(queries, *, mfs, config, etd, lamp, runways, standtimes,
        start_time, end_time, vocabularies, holiday_midnights, entity_snapshot_available_at):
    """Assemble un-imputed source-named features through corrected operations.

    Snapshot availability must be explicitly attested by the caller. Historical
    observations retain individual timestamps. This function does not read data
    files, infer vocabularies/fills or use model weights. Corrected availability,
    query-population and tie policies require consistent retraining. Missing
    actual-event timestamps exclude events rather than inventing availability.
    """
    _times(queries['timestamp'])
    start=pd.Timestamp(start_time);end=pd.Timestamp(end_time)
    snapshot=pd.Timestamp(entity_snapshot_available_at)
    if any(pd.isna(v) or v.tz is not None for v in (start,end,snapshot)) or start>end:
        raise ValueError('Valid naive context and snapshot times required')
    if len(queries) and (queries['timestamp'].min()<start or queries['timestamp'].max()>end or snapshot>queries['timestamp'].min()):
        raise ValueError('Queries must lie in context with an available entity snapshot')
    pairs=queries[['gufi','timestamp']].drop_duplicates().reset_index(drop=True)
    times=pairs[['timestamp']].drop_duplicates().reset_index(drop=True)
    query_ticks=_times(times['timestamp'])
    lexical=structured_identity_features(pairs['gufi'].to_numpy(),pd.DatetimeIndex(pairs['timestamp']),mfs['gufi'].to_numpy(),
        delimiter='.',minimum_parts=6,primary_part=0,category_part=2,reference_parts=[3,4],reference_format='%y%m%d%H%M',
        vocabularies={'primary':vocabularies['FLIGHT_IDS'],'numeric':vocabularies['FLIGHT_NUMBERS'],'category':vocabularies['AIRPORTS']},
        fallback='OTHER',seconds_per_unit=60)
    lexical.columns=['mfs_cat_flightid','mfs_cat_airline','mfs_cat_flightno','mfs_flightno_length',
        'mfs_cat_no_last','mfs_cat_arrival','mfs_cat_last','mfs_cat_sectolast','mfs_cat_hourdep','mfs_time_diff']
    attribute_map={'aircraft_engine_class':('mfs_cat_engineclass','ENGINE_CLASSES'),
        'aircraft_type':('mfs_cat_type','AIRCRAFT_TYPES'),'major_carrier':('mfs_cat_majorcarrier','CARRIER_TYPES'),
        'flight_type':('mfs_cat_flighttype','FLIGHT_TYPES'),'isdeparture':('mfs_departure',None)}
    attributes=unique_entity_features(pairs['gufi'].to_numpy(),mfs['gufi'].to_numpy(),mfs[list(attribute_map)],
        {key:vocabularies[rule] for key,(_,rule) in attribute_map.items() if rule},'OTHER')
    attributes.columns=[value[0] for value in attribute_map.values()]
    entity=pd.concat([pairs,lexical,attributes],axis=1)
    grid=pd.date_range(start,end,freq='5min')
    config_windows=np.array([6,12,24,36,48,120],dtype=np.int64)
    configuration=configuration_state_features(_times(config['timestamp']),
        config[['departure_runways','arrival_runways']].to_numpy(),_times(grid),vocabularies['RUNWAYS'],['R','L','C'],config_windows,5*MINUTE)
    columns={'timestamp':grid}
    for channel,(long_name,short_name) in enumerate([('departure','dep'),('arrival','arr')]):
        for vi,token in enumerate(vocabularies['RUNWAYS']):
            columns[f'config_{long_name}_cat_{token}']=configuration['categories'][:,channel,vi]
        columns[f'config_n_active_{short_name}']=configuration['item_counts'][:,channel]
        columns[f'config_change_{short_name}']=configuration['changes'][:,channel]
        for wi,width in enumerate(config_windows):columns[f'config_n_chang_{short_name}_last_{width}']=configuration['rolling_changes'][:,channel,wi]
    configuration_table=pd.DataFrame(columns).astype({key:float for key in columns if key!='timestamp'})
    configuration_table.loc[~configuration['state_valid'],configuration_table.columns!='timestamp']=np.nan
    windows=np.array([60,120,240,360,720,1440],dtype=np.int64)
    estimates=estimate_history_features(etd['gufi'].to_numpy(),_times(etd['timestamp']),
        _times(etd['departure_runway_estimated_time'],missing=True),etd['departure_runway_estimated_time'].notna().to_numpy(),
        pairs['gufi'].to_numpy(),_times(pairs['timestamp']),windows*MINUTE,15*MINUTE,MINUTE)
    estimate_columns={**{key:pairs[key] for key in pairs},'etd_est_change':estimates['change'],
        'etd_diff_withfirst':estimates['change_from_first'],'etd_time_till_est_dep':estimates['remaining']}
    for wi,width in enumerate(windows):
        for stat in ['sum','max','std']:estimate_columns[f'etd_roll_{width}_{stat}']=estimates[stat][:,wi]
    calendar=calendar_time_features(pd.DatetimeIndex(times['timestamp']),holiday_midnights)
    calendar_names=['mom_cat_hourofday','mom_cat_dayofweek','mom_cat_month','mom_dayofmonth','mom_cat_woy',
        'mom_cat_wom','mom_cat_isholiday','mom_days_until_holiday']
    calendar_table=pd.DataFrame({'timestamp':times['timestamp'],**dict(zip(calendar_names,calendar.values()))})
    forecast_values=lamp[WEATHER_FIELDS].copy()
    forecast_values['lightning_prob']=forecast_values['lightning_prob'].map({'L':0,'M':1,'N':2,'H':3})
    forecast_values['cloud']=forecast_values['cloud'].map({'OV':4,'BK':3,'CL':0,'FW':1,'SC':2}).fillna(3)
    forecast_values['precip']=forecast_values['precip'].astype(float)
    horizons=[1,3,6,9,12,24]
    forecasts=forecast_history_features(_times(lamp['timestamp']),_times(lamp['forecast_timestamp']),
        forecast_values.to_numpy(dtype=np.float64),query_ticks,history_window=360*MINUTE,
        lead_bands=[(None,210*MINUTE),(210*MINUTE,390*MINUTE)],
        contrast_leads=np.array([[30,60*h+30] for h in horizons],dtype=np.int64)*MINUTE,revision_lags=np.arange(1,7,dtype=np.int64))
    weather_columns={'timestamp':times['timestamp']}
    for fi,name in enumerate(WEATHER_FIELDS):
        for stat in ['mean','min','max']:weather_columns[f'temp_{name}_{stat}_last6h']=forecasts[stat][:,fi]
        for bi,suffix in enumerate(['next3','next36']):weather_columns[f'temp_{name}_{suffix}']=forecasts['bands'][:,bi,fi]
        weather_columns[f'temp_{name}']=forecasts['latest_valid'][:,fi]
        for lag in range(1,7):weather_columns[f'temp_change_{name}_{lag}']=forecasts['revisions'][:,lag-1,fi]
        for hi,horizon in enumerate(horizons):weather_columns[f'temp_futchange_{name}_{horizon}']=forecasts['contrasts'][:,hi,fi]
    event_windows=np.array([15,30,60,90,120,360,720,1440],dtype=np.int64)
    actual_columns=['departure_runway_actual_time','arrival_runway_actual_time']
    countable=np.column_stack([runways['gufi'].notna()&runways[category].notna()&runways[actual].notna()
        for category,actual in zip(['departure_runway_actual','arrival_runway_actual'],actual_columns)])
    counts=event_count_windows(_times(runways['timestamp']),np.column_stack([_times(runways[c],missing=True) for c in actual_columns]),
        countable,query_ticks,event_windows*MINUTE)
    count_columns={'timestamp':times['timestamp']}
    for wi,width in enumerate(event_windows):
        for ci,suffix in enumerate(['totdep','totarr']):count_columns[f'runways_{width}_{suffix}']=counts[:,ci,wi]
    events=standtimes.loc[standtimes['gufi'].notna()&standtimes['departure_stand_actual_time'].notna()]
    delay=event_delay_windows(events['gufi'].to_numpy(),_times(events['timestamp']),_times(events['departure_stand_actual_time']),
        etd['gufi'].to_numpy(),_times(etd['timestamp']),_times(etd['departure_runway_estimated_time'],missing=True),
        etd['departure_runway_estimated_time'].notna().to_numpy(),query_ticks,windows*MINUTE,15*MINUTE,MINUTE)
    delay_columns={'timestamp':times['timestamp']}
    for wi,hours in enumerate([1,2,4,6,12,24]):
        for name,position in [('count',0),('meandelay',2),('maxdelay',3)]:delay_columns[f'standtimes_{name}_{hours}']=delay[position][:,wi]
    return assemble_feature_tables(queries,[(entity,['gufi','timestamp']),(configuration_table,['timestamp']),
        (pd.DataFrame(estimate_columns),['gufi','timestamp']),(calendar_table,['timestamp']),
        (pd.DataFrame(weather_columns),['timestamp']),(pd.DataFrame(count_columns),['timestamp']),
        (pd.DataFrame(delay_columns),['timestamp'])])

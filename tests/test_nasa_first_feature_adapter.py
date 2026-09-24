"""Only independently constructed synthetic software-field fixtures."""
import numpy as np
import pandas as pd
import pytest

from sciona.nasa_first_feature_adapter import nasa_first_feature_tables, WEATHER_FIELDS


def fixture():
    origin=pd.Timestamp('2000-01-01')
    ids=['SYN1.SRC.DST.000101.0000.tail','SYN2.SRC.DST.000101.0000.tail']
    queries=pd.DataFrame({'gufi':[ids[0],ids[1],ids[0]],'timestamp':[origin,origin+pd.Timedelta(minutes=30),origin],
        'airport':['synthetic']*3},index=[8,3,8])
    mfs=pd.DataFrame({'gufi':ids,'aircraft_engine_class':['A']*2,'aircraft_type':['A']*2,
        'major_carrier':['A']*2,'flight_type':['A']*2,'isdeparture':[True,False]})
    config=pd.DataFrame({'timestamp':[origin,origin+pd.Timedelta(minutes=15)],'departure_runways':['07R','18L'],'arrival_runways':['18C','07']})
    etd=pd.DataFrame({'gufi':ids,'timestamp':[origin-pd.Timedelta(minutes=15)]*2,
        'departure_runway_estimated_time':[origin+pd.Timedelta(minutes=60),origin+pd.Timedelta(minutes=90)]})
    lamp=pd.DataFrame([{**{name:float(i) for i,name in enumerate(WEATHER_FIELDS)},
        'cloud':'CL','lightning_prob':'L','timestamp':origin+pd.Timedelta(minutes=issue),
        'forecast_timestamp':origin+pd.Timedelta(minutes=issue+lead)} for issue in [0,15] for lead in [0,30,90,210,390,570,750,1470]])
    runways=pd.DataFrame({'gufi':ids,'timestamp':[origin-pd.Timedelta(minutes=10),origin+pd.Timedelta(minutes=10)],
        'departure_runway_actual':['07','18'],'arrival_runway_actual':['18','07'],
        'departure_runway_actual_time':[origin-pd.Timedelta(minutes=10),origin+pd.Timedelta(minutes=10)],
        'arrival_runway_actual_time':[origin-pd.Timedelta(minutes=10),origin+pd.Timedelta(minutes=20)]})
    standtimes=runways[['gufi','timestamp']].copy()
    standtimes['departure_stand_actual_time']=runways['departure_runway_actual_time']
    options=dict(mfs=mfs,config=config,etd=etd,lamp=lamp,runways=runways,standtimes=standtimes,
        start_time=origin,end_time=origin+pd.Timedelta(minutes=30),entity_snapshot_available_at=origin,
        holiday_midnights=pd.DatetimeIndex(['2000-01-02','2000-02-01']),
        vocabularies={'FLIGHT_IDS':['SYN1','SYN2'],'FLIGHT_NUMBERS':['1','2'],'AIRPORTS':['DST'],
            'ENGINE_CLASSES':['A'],'AIRCRAFT_TYPES':['A'],'CARRIER_TYPES':['A'],'FLIGHT_TYPES':['A'],'RUNWAYS':['07','18']})
    return queries,options


def test_full_raw_feature_assembly_and_query_independence():
    queries,options=fixture();before={k:v.copy(deep=True) for k,v in options.items() if isinstance(v,pd.DataFrame)}
    result=nasa_first_feature_tables(queries,**options)
    assert result.shape==(3,263)
    pd.testing.assert_frame_equal(result[queries.columns],queries)
    np.testing.assert_array_equal(result['etd_time_till_est_dep'],[60,60,60])
    np.testing.assert_array_equal(result['runways_15_totdep'],[1,0,1])
    for i in range(len(queries)):
        pd.testing.assert_frame_equal(result.iloc[[i]],nasa_first_feature_tables(queries.iloc[[i]],**options))
    for k,v in before.items():pd.testing.assert_frame_equal(options[k],v)


def test_future_observation_perturbation_isolation():
    queries,options=fixture();expected=nasa_first_feature_tables(queries.iloc[[0]],**options)
    options['config'].loc[1,'departure_runways']='07C'
    options['lamp'].loc[options['lamp']['timestamp']>queries['timestamp'].iloc[0],'temperature']+=1000
    options['runways'].loc[1,'departure_runway_actual_time']+=pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(expected,nasa_first_feature_tables(queries.iloc[[0]],**options))


def test_unavailable_entity_snapshot_rejected():
    queries,options=fixture();options['entity_snapshot_available_at']+=pd.Timedelta(minutes=1)
    with pytest.raises(ValueError,match='available entity snapshot'):nasa_first_feature_tables(queries,**options)


def test_unknown_weather_categories_match_declared_source_mapping():
    queries,options=fixture();options['lamp']['cloud']='unknown';options['lamp']['lightning_prob']='unknown'
    result=nasa_first_feature_tables(queries,**options)
    np.testing.assert_array_equal(result['temp_cloud_next3'],3)
    assert result['temp_lightning_prob_next3'].isna().all()

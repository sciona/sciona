"""Independent synthetic raw inputs for native workflow qualification."""
import numpy as np
import pandas as pd

from sciona.model_feature_policy import prepare_model_features
from sciona.nasa_first_feature_adapter import nasa_first_feature_tables, WEATHER_FIELDS


def population_table(population, time_start):
    rng=np.random.default_rng(2884+population)
    count=192;repetitions=128
    times=pd.date_range(time_start,periods=count,freq='15min')
    ids=np.array([f'SYN{i%8}.SRC.DST.{stamp:%y%m%d}.{stamp:%H%M}.tail' for i,stamp in enumerate(times)])
    queries=pd.DataFrame({'gufi':ids,'timestamp':times})
    mfs=pd.DataFrame({'gufi':ids,'aircraft_engine_class':rng.choice(['A','B'],count),
        'aircraft_type':rng.choice(['A','B'],count),'major_carrier':rng.choice(['A','B'],count),
        'flight_type':rng.choice(['A','B'],count),'isdeparture':rng.integers(0,2,count)})
    config=pd.DataFrame({'timestamp':times[::4],
        'departure_runways':rng.choice(['07R','18L'],len(times[::4])),
        'arrival_runways':rng.choice(['18C','07'],len(times[::4]))})
    baseline=rng.integers(30,120,count)
    etd=pd.DataFrame({'gufi':np.repeat(ids,2),
        'timestamp':np.repeat(times.to_numpy(),2)-np.tile(pd.to_timedelta([30,15],unit='min').to_numpy(),count),
        'departure_runway_estimated_time':np.repeat((times+pd.to_timedelta(baseline,unit='min')).to_numpy(),2)})
    rows=[]
    for stamp in times[::4]:
        for lead in [0,30,90,210,270,390,570,750,1470]:
            rows.append({'timestamp':stamp,'forecast_timestamp':stamp+pd.Timedelta(minutes=lead),
                **{name:float(rng.uniform(0,100)) for name in WEATHER_FIELDS},
                'cloud':rng.choice(['CL','OV']),'lightning_prob':rng.choice(['L','H'])})
    lamp=pd.DataFrame(rows)
    event_times=times-pd.Timedelta(minutes=5)
    runways=pd.DataFrame({'gufi':ids,'timestamp':event_times,'departure_runway_actual':['07']*count,
        'arrival_runway_actual':['18']*count,'departure_runway_actual_time':event_times,
        'arrival_runway_actual_time':event_times})
    standtimes=pd.DataFrame({'gufi':ids,'timestamp':event_times,'departure_stand_actual_time':event_times})
    raw=nasa_first_feature_tables(queries,mfs=mfs,config=config,etd=etd,lamp=lamp,runways=runways,standtimes=standtimes,
        start_time=times[0],end_time=times[-1],entity_snapshot_available_at=times[0]-pd.Timedelta(days=1),
        holiday_midnights=pd.DatetimeIndex([times[-1].normalize()+pd.Timedelta(days=1)]),
        vocabularies={'FLIGHT_IDS':[f'SYN{i}' for i in range(8)],'FLIGHT_NUMBERS':[str(i) for i in range(8)],
            'AIRPORTS':['DST'],'ENGINE_CLASSES':['A','B'],'AIRCRAFT_TYPES':['A','B'],
            'CARRIER_TYPES':['A','B'],'FLIGHT_TYPES':['A','B'],'RUNWAYS':['07','18']})
    order=sorted(set(raw)-set(queries))
    categories={column:'OTHER' for column in order if '_cat_' in column}
    numeric={column:0 for column in order if column not in categories}
    prepared=prepare_model_features(raw,order,numeric,categories)
    # Repeated synthetic queries supply the source trainer's large leaf setting.
    # These repetitions are not independent observations or performance evidence.
    expanded=np.repeat(np.arange(count),repetitions)
    table=pd.concat([queries,prepared],axis=1).iloc[expanded].reset_index(drop=True)
    table['minutes_until_pushback']=100+rng.uniform(1,100,len(table))
    return table,dict(unique_raw_queries=count,synthetic_repetitions_per_query=repetitions,
        model_rows=len(table),feature_columns=len(order),categorical_features=len(categories),
        raw_features_computed=True,policy_inferred_from_query_values=False)

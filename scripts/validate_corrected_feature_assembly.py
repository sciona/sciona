"""Synthetic composition of all seven corrected feature families and imputation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sciona.calendar_time_features import calendar_time_features
from sciona.configuration_state_features import configuration_state_features
from sciona.estimate_history_features import estimate_history_features
from sciona.event_count_windows import event_count_windows
from sciona.event_delay_windows import event_delay_windows
from sciona.feature_table_assembly import assemble_feature_tables
from sciona.fixed_feature_imputation import fixed_integer_features
from sciona.forecast_history_features import forecast_history_features
from sciona.structured_identity_features import structured_identity_features
from sciona.unique_entity_features import unique_entity_features

ROOT=Path(__file__).resolve().parents[1]


def frame(prefix, keys, outputs):
    columns={}
    for name,values in outputs.items():
        flat=np.asarray(values).reshape(len(keys),-1)
        for column in range(flat.shape[1]):columns[f'{prefix}_{name}_{column}']=flat[:,column]
    return pd.concat([keys.reset_index(drop=True),pd.DataFrame(columns)],axis=1)


def compose(queries):
    origin=pd.Timestamp('2000-01-01')
    identities=np.array(['M1.line.000101.0000.tail','M2.line.000101.0000.tail'])
    pairs=queries[['entity','tick']].drop_duplicates().reset_index(drop=True)
    grid=np.array([0,15,30],dtype=np.int64)
    time_keys=pd.DataFrame({'tick':grid})
    windows=np.array([60,120,240,360,720,1440],dtype=np.int64)
    lexical=structured_identity_features(pairs['entity'].to_numpy(),
        pd.DatetimeIndex(origin+pd.to_timedelta(pairs['tick'],unit='min')),identities,
        delimiter='.',minimum_parts=5,primary_part=0,category_part=1,reference_parts=[2,3],
        reference_format='%y%m%d%H%M',vocabularies={'primary':['M1','M2'],'numeric':['1','2'],'category':['line']},
        fallback='OTHER',seconds_per_unit=60)
    attributes=unique_entity_features(pairs['entity'].to_numpy(),identities,
        pd.DataFrame({'mode':['on','off'],'size':[4.,8.]}),{'mode':['on','off']},'OTHER')
    entity=pd.concat([pairs,lexical.add_prefix('entity_'),attributes.add_prefix('entity_')],axis=1)
    config=configuration_state_features(np.array([0,15],dtype=np.int64),np.array([['unitON'],['unitOFF']]),
        grid,['unit'],['ON','OFF'],np.array([2],dtype=np.int64),15)
    estimates=estimate_history_features(identities,np.array([-15,-15],dtype=np.int64),
        np.array([60,90],dtype=np.int64),np.ones(2,dtype=bool),pairs['entity'].to_numpy(),
        pairs['tick'].to_numpy(dtype=np.int64),windows,15,1)
    calendar=calendar_time_features(pd.DatetimeIndex(origin+pd.to_timedelta(grid,unit='min')),
        pd.DatetimeIndex(['2000-01-02','2000-02-01']))
    forecasts=forecast_history_features(np.array([0,0,15,15],dtype=np.int64),
        np.array([0,30,15,45],dtype=np.int64),np.arange(36,dtype=np.float64).reshape(4,9),grid,
        history_window=360,lead_bands=[(None,210),(210,390)],
        contrast_leads=np.array([[30,60*h+30] for h in [1,3,6,9,12,24]],dtype=np.int64),
        revision_lags=np.arange(1,7,dtype=np.int64))
    counts=event_count_windows(np.array([-10,10],dtype=np.int64),
        np.array([[-10,-10],[10,20]],dtype=np.int64),np.ones((2,2),dtype=bool),grid,
        np.array([15,30,60,90,120,360,720,1440],dtype=np.int64))
    delays=event_delay_windows(identities,np.array([-10,10],dtype=np.int64),np.array([-10,10],dtype=np.int64),
        identities,np.array([-20,-20],dtype=np.int64),np.array([-15,5],dtype=np.int64),np.ones(2,dtype=bool),
        grid,windows,15,1)
    tables=[(entity,['entity','tick']),
        (frame('configuration',time_keys,config),['tick']),
        (frame('estimate',pairs,estimates),['entity','tick']),
        (frame('calendar',time_keys,calendar),['tick']),
        (frame('forecast',time_keys,forecasts),['tick']),
        (frame('event_count',time_keys,{'count':counts}),['tick']),
        (frame('event_delay',time_keys,dict(zip(['count','valid_count','mean','max'],delays))),['tick'])]
    assembled=assemble_feature_tables(queries,tables)
    numerical=[c for c in assembled if c not in queries and pd.api.types.is_numeric_dtype(assembled[c])]
    # Explicit synthetic policy; this does not infer production fill values.
    fills=np.zeros(len(numerical),dtype=np.float64)
    assembled[numerical]=fixed_integer_features(assembled[numerical].to_numpy(dtype=np.float64),fills)
    return assembled,numerical


def validate():
    queries=pd.DataFrame({'entity':['M1.line.000101.0000.tail','M2.line.000101.0000.tail','M1.line.000101.0000.tail'],
        'tick':[0,30,0]},index=[8,3,8])
    combined,numerical=compose(queries)
    pd.testing.assert_frame_equal(combined[queries.columns],queries)
    for i in range(len(queries)):
        solo,solo_numeric=compose(queries.iloc[[i]])
        assert solo_numeric==numerical
        pd.testing.assert_frame_equal(combined.iloc[[i]],solo)
    order=[2,0,1]
    reordered,_=compose(queries.iloc[order])
    pd.testing.assert_frame_equal(combined.iloc[order],reordered)
    assert combined[numerical].notna().all().all()
    assert all(dtype==np.dtype('int16') for dtype in combined[numerical].dtypes)
    modules=['calendar_time_features','configuration_state_features','estimate_history_features','event_count_windows',
        'event_delay_windows','feature_table_assembly','fixed_feature_imputation','forecast_history_features',
        'structured_identity_features','unique_entity_features']
    paths=[f'sciona/{name}.py' for name in modules]+['scripts/validate_corrected_feature_assembly.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,feature_families=7,
        query_rows=len(queries),numeric_feature_columns=len(numerical),total_feature_columns=len(combined.columns)-len(queries.columns),
        independent_query_compositions=3,query_order_and_duplicates_preserved=True,fixed_imputation=True,
        limitations=['Synthetic generic composition, not the complete source-specific raw-field adapter or training workflow.',
            'Production calendar/vocabulary/availability and fill policies, native retraining and catalog qualification remain pending.'],
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_nasa_first_corrected_feature_assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['passed','feature_families','numeric_feature_columns','total_feature_columns','approved']}))

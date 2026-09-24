"""Bind source feature policies to native models, without storing raw records."""
import json
import numpy as np
import pandas as pd
from sciona.model_feature_policy import prepare_model_features

FORMAT='sciona.first-place.feature-policy.v1'


def _calendar(value):
    times=pd.DatetimeIndex(value)
    if times.empty or times.hasnans or times.tz is not None or not times.is_unique or not times.is_monotonic_increasing or not times.equals(times.normalize()):
        raise ValueError('Ordered unique naive holiday midnights required')
    return [time.isoformat() for time in times]


def capture_policy(entry):
    columns=entry['feature_columns'];numeric=entry['numeric_fills'];categorical=entry['categorical_fills']
    if not isinstance(numeric,dict) or not isinstance(categorical,dict):
        raise ValueError('Explicit fixed feature fills required')
    probe=pd.DataFrame({name:[np.nan if name in numeric else None] for name in columns})
    prepare_model_features(probe,columns,numeric,categorical,integer_dtype='int16')
    vocabularies=entry['raw_options']['vocabularies']
    if not isinstance(vocabularies,dict) or any(not isinstance(name,str) or not isinstance(values,list)
        or any(not isinstance(value,str) for value in values) or len(set(values))!=len(values) for name,values in vocabularies.items()):
        raise ValueError('Explicit named unique string vocabularies required')
    required={'FLIGHT_IDS','FLIGHT_NUMBERS','AIRPORTS','ENGINE_CLASSES','AIRCRAFT_TYPES','CARRIER_TYPES','FLIGHT_TYPES','RUNWAYS'}
    if not required.issubset(vocabularies):
        raise ValueError('Complete source vocabulary policies required')
    result=dict(feature_columns=columns,numeric_fills=numeric,categorical_fills=categorical,
        integer_dtype='int16',vocabularies=vocabularies,holiday_midnights=_calendar(entry['raw_options']['holiday_midnights']))
    return json.loads(json.dumps(result,allow_nan=False))


def validate_model_policy(slots,policy):
    columns=policy['feature_columns'];categories=policy['categorical_fills']
    if set(slots)!={0,1,2,'global_model'} or slots[0] is not slots[2]:
        raise ValueError('Complete source model slots and residual alias required')
    for slot in (0,1,2,'global_model'):
        expected=columns if slot!='global_model' else [name for name in columns if 'mfs' in name or 'etd' in name]+['feat_cat_airport']
        indices=[i for i,name in enumerate(expected) if name in categories or name=='feat_cat_airport']
        if list(slots[slot].feature_names_)!=expected or list(slots[slot].get_cat_feature_indices())!=indices:
            raise ValueError('Native model schema differs from bound feature policy')


def capture_bank_policy(bank,raw_populations):
    if set(bank)!={name.upper() for name in raw_populations}:
        raise ValueError('Model and training policy populations differ')
    policies={}
    for name,entry in raw_populations.items():
        policy=capture_policy(entry)
        validate_model_policy(bank[name.upper()],policy)
        policies[name.upper()]=policy
    return dict(format=FORMAT,target_unit='minutes',populations=policies)


def bound_prediction_inputs(bank,policy,population,raw_options):
    if set(policy)!={'format','target_unit','populations'} or policy['format']!=FORMAT or policy['target_unit']!='minutes':
        raise ValueError('Complete source feature-policy state required')
    if set(bank)!=set(policy['populations']) or population.upper() not in bank:
        raise ValueError('Prediction population absent from bound model policy')
    selected=policy['populations'][population.upper()]
    validated=capture_policy(dict(feature_columns=selected['feature_columns'],numeric_fills=selected['numeric_fills'],
        categorical_fills=selected['categorical_fills'],raw_options=selected))
    if validated!=selected:raise ValueError('Stored feature policy differs from supported policy')
    slots=bank[population.upper()];validate_model_policy(slots,selected)
    if 'vocabularies' in raw_options and raw_options['vocabularies']!=selected['vocabularies']:
        raise ValueError('Inference vocabulary differs from training policy')
    if 'holiday_midnights' in raw_options and _calendar(raw_options['holiday_midnights'])!=selected['holiday_midnights']:
        raise ValueError('Inference calendar differs from training policy')
    options=dict(raw_options,vocabularies=selected['vocabularies'],holiday_midnights=pd.DatetimeIndex(selected['holiday_midnights']))
    return slots,options,selected['feature_columns'],selected['numeric_fills'],selected['categorical_fills']

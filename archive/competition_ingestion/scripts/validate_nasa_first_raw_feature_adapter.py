"""Pinned-software coverage of the corrected raw-field adapter, synthetic only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import runpy
import warnings

import numpy as np
import pandas as pd

from sciona.nasa_first_feature_adapter import nasa_first_feature_tables
from sciona.model_feature_policy import prepare_model_features

ROOT=Path(__file__).resolve().parents[1]
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/'1st Place/Phase 1/submission/utilities.py').read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned software differs')
    fixture=runpy.run_path(str(ROOT/'tests/test_nasa_first_feature_adapter.py'))['fixture']
    queries,options=fixture()
    class Calendar:
        def holidays(self,**kwargs):return options['holiday_midnights']
    names=['extract_mfs_features','extract_config_features','extract_etd_features','extract_moment_features',
        'extract_weather_features','extract_runway_features','extract_standtime_features']
    nodes=[node for node in ast.parse(data).body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(nodes)!=7:raise ValueError('Complete feature source not found')
    namespace={'np':np,'pd':pd,'re':re,'calendar':Calendar,'START_HOLIDAY':'2000-01-01','END_HOLIDAY':'2000-02-01',**options['vocabularies']}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_features','exec'),namespace)
    pairs=queries[['gufi','timestamp']].drop_duplicates().reset_index(drop=True)
    context=[str(options['start_time']),str(options['end_time'])]
    arguments=[(pairs,options['mfs']),(options['config'],*context),(pairs,options['etd']),
        (pairs,),(options['lamp'],*context),(options['runways'],*context),(pairs,options['standtimes'],options['etd'])]
    actual=nasa_first_feature_tables(queries,**options)
    covered=set();compatible_values=0
    for name,args in zip(names,arguments):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            expected=namespace[name](*[arg.copy(deep=True) if isinstance(arg,pd.DataFrame) else arg for arg in args])
        keys=['gufi','timestamp'] if 'gufi' in expected else ['timestamp']
        features=set(expected)-set(keys)
        if covered&features:raise ValueError('Unexpected overlapping source features')
        covered.update(features)
        if name=='extract_weather_features':continue  # Explicit corrections checked in dedicated forecast evidence.
        projected=queries[keys].merge(expected,on=keys,how='left',validate='many_to_one')
        for column in features:
            pd.testing.assert_series_equal(actual[column].reset_index(drop=True),projected[column],check_dtype=False,check_exact=False,rtol=1e-14,atol=1e-14)
            compatible_values+=len(queries)
    if covered!=set(actual)-set(queries):raise ValueError('Raw adapter feature-column coverage differs')
    # The source trainer declares categorical features by this name marker.
    # Fill constants below are an explicit synthetic policy, never fitted to rows.
    order=sorted(covered)
    categorical={column:'OTHER' for column in order if '_cat_' in column}
    numeric={column:0 for column in order if column not in categorical}
    prepared=prepare_model_features(actual,order,numeric,categorical)
    if prepared.isna().any().any():raise ValueError('Model-input policy left missing values')
    restored=json.loads(json.dumps(dict(order=order,numeric=numeric,categorical=categorical)))
    for i in range(len(queries)):
        solo=nasa_first_feature_tables(queries.iloc[[i]],**options)
        pd.testing.assert_frame_equal(prepared.iloc[[i]],prepare_model_features(solo,restored['order'],restored['numeric'],restored['categorical']))
    paths=['sciona/nasa_first_feature_adapter.py','tests/test_nasa_first_feature_adapter.py',
        'sciona/model_feature_policy.py','tests/test_model_feature_policy.py','scripts/validate_nasa_first_raw_feature_adapter.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        source_feature_families=7,source_feature_columns=len(covered),compatible_nonweather_values=compatible_values,
        prepared_model_columns=len(prepared.columns),categorical_model_columns=len(categorical),
        fixed_policy_roundtrip=True,prepared_batch_independence=True,
        limitations=['Synthetic runtime vocabularies and calendar; original production configuration not qualified.',
            'Weather corrections have separate component evidence; this validation checks its complete source column coverage.',
            'Fixed synthetic model-input policy is checked; production policy qualification, native retraining, fallback composition and version-bound catalog qualification remain required.'],
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_raw_feature_adapter.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','source_feature_columns','compatible_nonweather_values','approved']}))

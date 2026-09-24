"""Execute pinned forecast features on synthetic issue/valid-time grids."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sciona.forecast_history_features import forecast_history_features

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned source differs')
    function=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='extract_weather_features')
    sort=next(n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sort_values' and n.args and isinstance(n.args[0],ast.List))
    issue,valid=ast.literal_eval(sort.args[0])
    feature_names=ast.literal_eval(next(n.value for n in function.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='temp_feats'))
    mappings={}
    for node in ast.walk(function):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='map' and node.args and isinstance(node.args[0],ast.Dict):
            mappings[node.func.value.slice.value]=ast.literal_eval(node.args[0])
    namespace={'pd':pd,'np':np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_weather_features','exec'),namespace)
    origin=pd.Timestamp('2001-01-01')
    leads=[.5,1.5,3.5,4.5,6.5,9.5,12.5,24.5]
    rows=[]
    for minute in [60,120]:
        for lead in leads:
            row={issue:origin+pd.Timedelta(minutes=minute),valid:origin+pd.Timedelta(minutes=minute)+pd.Timedelta(hours=lead)}
            for j,name in enumerate(feature_names):row[name]=minute+lead*10+j
            for name,mapping in mappings.items():row[name]=next(iter(mapping))
            rows.append(row)
    frame=pd.DataFrame(rows)
    def run(frame):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            return namespace[function.name](frame.copy(),str(origin),str(origin+pd.Timedelta(hours=3)))
    result=run(frame)
    numeric=feature_names[0]
    perturbed=frame.copy();perturbed.loc[perturbed[issue]==origin+pd.Timedelta(hours=1),numeric]+=1000
    changed=run(perturbed)
    first=result.loc[result[issue]==origin].iloc[0]
    other=changed.loc[changed[issue]==origin].iloc[0]
    differences=[c for c in result if c!=issue and not np.isclose(first[c],other[c],equal_nan=True)]
    if not differences:raise ValueError('Future issue availability counterexample disappeared')
    horizons=[1,3,6,9,12,24]
    checks=0
    row=result.loc[result[issue]==origin+pd.Timedelta(hours=2)].iloc[0]
    for horizon in horizons:
        column='temp_futchange_'+numeric+'_'+str(horizon)
        np.testing.assert_allclose(row[column],-10*horizon,rtol=0,atol=0);checks+=1
    # Forecast lead intervals are asymmetric: <=3.5 includes negative leads;
    # the second interval is strictly >3.5 and <=6.5.
    for suffix,chosen in [('next3',[v for v in leads if v<=3.5]),('next36',[v for v in leads if 3.5<v<=6.5])]:
        expected=np.mean([120+10*v for v in chosen])
        np.testing.assert_allclose(row['temp_'+numeric+'_'+suffix],expected,rtol=0,atol=0);checks+=1
    converted=frame[feature_names].copy()
    for name,mapping in mappings.items():converted[name]=converted[name].map(mapping)
    issued=np.array([(stamp-origin).total_seconds()/60 for stamp in frame[issue]],dtype=np.int64)
    valid_times=np.array([(stamp-origin).total_seconds()/60 for stamp in frame[valid]],dtype=np.int64)
    matrix=converted.to_numpy(dtype=np.float64)
    query=np.array([0,120],dtype=np.int64)
    controls=dict(history_window=360,lead_bands=[(None,210),(210,390)],
        contrast_leads=np.array([[30,60*h+30] for h in horizons],dtype=np.int64),revision_lags=np.arange(1,7,dtype=np.int64))
    corrected=forecast_history_features(issued,valid_times,matrix,query,**controls)
    corrected_checks=0
    for fi,name in enumerate(feature_names):
        for hi,horizon in enumerate(horizons):
            np.testing.assert_allclose(corrected['contrasts'][1,hi,fi],row['temp_futchange_'+name+'_'+str(horizon)],rtol=0,atol=0)
            corrected_checks+=1
        for bi,suffix in enumerate(['next3','next36']):
            np.testing.assert_allclose(corrected['bands'][1,bi,fi],row['temp_'+name+'_'+suffix],rtol=0,atol=0)
            corrected_checks+=1
    if corrected['issue_available'][0]:raise ValueError('Future issue became available early')
    for key,value in corrected.items():
        if key!='issue_available' and not np.isnan(value[0]).all():raise ValueError('Unavailable forecast was populated')
    changed_matrix=matrix.copy();changed_matrix[issued==60,0]+=1000
    corrected_changed=forecast_history_features(issued,valid_times,changed_matrix,query,**controls)
    for key in corrected:np.testing.assert_array_equal(corrected[key][0],corrected_changed[key][0])
    # Dense shared valid-time coverage makes all six revision lags observable.
    dense_rows=[]
    for minute in range(0,121,15):
        for lead in range(-120,1471,15):
            entry={issue:origin+pd.Timedelta(minutes=minute),valid:origin+pd.Timedelta(minutes=minute+lead)}
            for fi,name in enumerate(feature_names):entry[name]=minute+lead/6+fi
            for name,mapping in mappings.items():entry[name]=next(iter(mapping))
            dense_rows.append(entry)
    dense=pd.DataFrame(dense_rows)
    dense_source=run(dense)
    dense_row=dense_source.loc[dense_source[issue]==origin+pd.Timedelta(minutes=120)].iloc[0]
    dense_values=dense[feature_names].copy()
    for name,mapping in mappings.items():dense_values[name]=dense_values[name].map(mapping)
    dense_corrected=forecast_history_features(
        np.array([(stamp-origin).total_seconds()/60 for stamp in dense[issue]],dtype=np.int64),
        np.array([(stamp-origin).total_seconds()/60 for stamp in dense[valid]],dtype=np.int64),
        dense_values.to_numpy(dtype=np.float64),np.array([120],dtype=np.int64),**controls)
    complete_checks=0
    for fi,name in enumerate(feature_names):
        comparisons=[('temp_'+name+'_'+stat+'_last6h',dense_corrected[stat][0,fi]) for stat in ['mean','min','max']]
        comparisons += [('temp_'+name+'_'+suffix,dense_corrected['bands'][0,bi,fi]) for bi,suffix in enumerate(['next3','next36'])]
        comparisons += [('temp_'+name,dense_corrected['latest_valid'][0,fi])]
        comparisons += [('temp_change_'+name+'_'+str(lag),dense_corrected['revisions'][0,lag-1,fi]) for lag in range(1,7)]
        comparisons += [('temp_futchange_'+name+'_'+str(horizon),dense_corrected['contrasts'][0,hi,fi]) for hi,horizon in enumerate(horizons)]
        for column,value in comparisons:
            np.testing.assert_allclose(value,dense_row[column],rtol=1e-14,atol=1e-14)
            complete_checks+=1
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        synthetic_forecast_rows=len(frame),grid_rows=len(result),feature_columns=len(result.columns)-1,
        independent_lead_and_difference_checks=checks,future_issue_changed_earlier_features=len(differences),
        corrected_lead_feature_comparisons=corrected_checks,corrected_future_issue_isolation=True,
        corrected_dense_feature_comparisons=complete_checks,
        findings=['Boundary padding and backward fill allow a later forecast issue to populate earlier grid points when no prior issue exists.',
            'Future-change features subtract a longer-lead forecast from the half-hour forecast for the same issue; they are forecast contrasts, not observed future outcomes.',
            'Near-term lead selection is <=3.5 hours without a lower bound; the following band is (3.5,6.5].',
            'Historical revisions group by forecast valid time, so issue time and valid time must remain separate contracts.'],
        requirements=['Select only forecasts issued by each query before computing features; preserve valid-time/lead semantics independently.',
            'Define missing prior-issue and missing-horizon policies without backward filling from unavailable issues.',
            'Retain categorical mapping and unknown-value behavior explicitly in the domain adapter.'],
        limitations=['Synthetic software comparison with two issue times; not a full forecast adapter or original-data leakage claim.',
            'Corrected history windows advance at each query, do not insert padding observations and do not fill missing horizons/revisions from other issues; retraining is required.',
            'Dense historical/revision parity is synthetic; categorical unknown policies, full training/inference composition and publication remain outstanding.'],
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['sciona/forecast_history_features.py','tests/test_forecast_history_features.py']},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_weather_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','feature_columns','independent_lead_and_difference_checks','future_issue_changed_earlier_features','approved']}))

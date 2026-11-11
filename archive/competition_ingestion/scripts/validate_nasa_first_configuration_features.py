"""Synthetic configuration availability and event-count boundary audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.fixed_window_statistics import fixed_window_statistics
from sciona.asof_indices import latest_available_indices
from sciona.configuration_state_features import configuration_state_features
from sciona.event_count_windows import event_count_windows

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned software differs')
    functions={n.name:n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef)}
    config=functions['extract_config_features'];runway=functions['extract_runway_features']
    time=next(n.args[0].value for n in ast.walk(config) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sort_values')
    drop=next(n for n in ast.walk(config) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='drop')
    categories=ast.literal_eval(next(k.value for k in drop.keywords if k.arg=='columns'))
    namespace={'pd':pd,'np':np,'RUNWAYS':['07','18']}
    exec(compile(ast.Module(body=[config,runway],type_ignores=[]),'pinned_configuration','exec'),namespace)
    origin=pd.Timestamp('2001-01-01')
    raw=pd.DataFrame({time:[origin+pd.Timedelta(minutes=10),origin+pd.Timedelta(minutes=20)],categories[0]:['07R','18L'],categories[1]:['18C','07']})
    result=namespace[config.name](raw.copy(),str(origin),str(origin+pd.Timedelta(minutes=30)))
    altered=raw.copy();altered.loc[0,categories]=['18L','07R']
    changed=namespace[config.name](altered,str(origin),str(origin+pd.Timedelta(minutes=30)))
    first=result.loc[result[time]==origin].iloc[0]
    modified=changed.loc[changed[time]==origin].iloc[0]
    unavailable_changes=sum(first[c]!=modified[c] for c in result.columns if c!=time)
    if unavailable_changes==0:raise ValueError('Future configuration counterexample not reproduced')
    indices,available=latest_available_indices(np.array([10,20],dtype=np.int64),np.arange(0,31,5,dtype=np.int64))
    np.testing.assert_array_equal(indices,[-1,-1,0,0,1,1,1])
    np.testing.assert_array_equal(available,[False,False,True,True,True,True,True])
    widths=np.array([6,12,24,36,48,120],dtype=np.int64)
    grid=np.arange(0,721,5,dtype=np.int64)
    observed=np.array([0,20,125,355,610],dtype=np.int64)
    states=np.array([['07R','18C'],['18L','07'],['07L,07R',''],['107R','18'],['','07C']])
    compatible=pd.DataFrame({time:origin+pd.to_timedelta(observed,unit='min'),
        categories[0]:states[:,0],categories[1]:states[:,1]})
    source_result=namespace[config.name](compatible,str(origin),str(origin+pd.Timedelta(minutes=720)))
    corrected=configuration_state_features(observed,states,grid,['07','18'],['R','L','C'],widths,5)
    source_comparisons=0
    for channel,(long_name,short_name) in enumerate([('departure','dep'),('arrival','arr')]):
        pairs=[(f'config_{long_name}_cat_{token}',corrected['categories'][:,channel,k]) for k,token in enumerate(['07','18'])]
        pairs += [(f'config_n_active_{short_name}',corrected['item_counts'][:,channel]),
                  (f'config_change_{short_name}',corrected['changes'][:,channel])]
        pairs += [(f'config_n_chang_{short_name}_last_{width}',corrected['rolling_changes'][:,channel,k]) for k,width in enumerate(widths)]
        for column,values in pairs:
            np.testing.assert_array_equal(source_result[column].to_numpy(),values)
            source_comparisons+=len(values)
    # A future state is unavailable, rather than backward-filled into early rows.
    early_grid=np.arange(0,31,5,dtype=np.int64)
    corrected_early=configuration_state_features(np.array([10,20],dtype=np.int64),raw[categories].to_numpy(),early_grid,['07','18'],['R','L','C'],widths,5)
    changed_early=configuration_state_features(np.array([10,20],dtype=np.int64),altered[categories].to_numpy(),early_grid,['07','18'],['R','L','C'],widths,5)
    for key in corrected_early:
        np.testing.assert_array_equal(corrected_early[key][:2],changed_early[key][:2])
    np.testing.assert_array_equal(corrected_early['state_valid'],available)
    # Nested token classifier intentionally uses substring matching and priority.
    process=next(n for n in config.body if isinstance(n,ast.FunctionDef))
    exec(compile(ast.Module(body=[process],type_ignores=[]),'pinned_category_rule','exec'),namespace)
    cases=[('', '07',0),('07R','07',1),('07L','07',2),('07C','07',3),('07','07',4),('107R','07',1),('07L,07R','07',1)]
    for text,key,expected in cases:
        if namespace['process'](text,key)!=expected:raise ValueError('Token classifier differs')
    drop=next(n for n in ast.walk(runway) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='drop')
    columns=ast.literal_eval(next(k.value for k in drop.keywords if k.arg=='columns'))
    identity,departure,departure_actual,arrival,arrival_actual,_=columns
    windows=np.array([15,30,60,90,120,360,720,1440],dtype=np.int64)
    comparisons=0;tie_mismatches=0;corrected_count_comparisons=0
    for tied in [False,True]:
        count=100
        offsets=np.zeros(count,dtype=np.int64) if tied else -np.arange(1,count+1,dtype=np.int64)
        history=pd.DataFrame({identity:['synthetic_'+str(i) for i in range(count)],time:[origin+pd.Timedelta(minutes=int(v)) for v in offsets],
            departure:['07']*count,arrival:['18']*count,
            departure_actual:[origin+pd.Timedelta(minutes=int(v)) for v in offsets],arrival_actual:[origin+pd.Timedelta(minutes=int(v)) for v in offsets]})
        actual=namespace[runway.name](history,str(origin),str(origin+pd.Timedelta(minutes=30)))
        query=np.array([0,15,30],dtype=np.int64)
        expected=fixed_window_statistics(offsets,offsets,offsets,np.zeros(count),query,windows)[0]
        corrected_counts=event_count_windows(offsets,np.column_stack([offsets,offsets]),
            np.ones((count,2),dtype=bool),query,windows)
        # Independent integer membership oracle includes every available tied row.
        oracle=np.array([[sum(int(q)-int(w)<int(t)<=int(q) for t in offsets) for w in windows] for q in query])
        for channel in range(2):
            np.testing.assert_array_equal(corrected_counts[:,channel],oracle)
            corrected_count_comparisons+=oracle.size
        for qi in range(3):
            for wi,width in enumerate(windows):
                for suffix in ['totdep','totarr']:
                    column=next(c for c in actual if c.endswith('_'+str(width)+'_'+suffix))
                    value=actual.iloc[qi][column]
                    if tied:tie_mismatches+=int(value!=expected[qi,wi])
                    elif value!=expected[qi,wi]:raise ValueError('Distinct-time rolling count differs')
                    comparisons+=1
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        configuration_grid_rows=len(result),corrected_asof_grid_points=len(indices),future_configuration_feature_changes=int(unavailable_changes),category_cases=len(cases),
        corrected_configuration_value_comparisons=source_comparisons,
        corrected_future_state_invariance=True,
        event_count_comparisons=comparisons,tied_timestamp_count_mismatches=tie_mismatches,
        corrected_event_count_comparisons=corrected_count_comparisons,
        findings=['Configuration boundary padding backward-fills from a later observation; changing that observation changes earlier features.',
            'Category encoding uses substring membership and R-before-L-before-C priority, not exact token matching.',
            'Event counts use observation timestamps and left-open/right-closed windows; marker rows at tied timestamps can depend on row order.'],
        requirements=['Define prior-state availability at the configuration-grid start rather than borrowing a future observation.',
            'Retain or explicitly correct substring encoding with a declared vocabulary; injected synthetic tokens do not qualify the original configuration.',
            'Use explicit timestamp-window membership for event counts and enforce actual-event availability before joining query rows.'],
        limitations=['Synthetic configuration vocabulary and software-only inputs; original configuration and empirical source records are not evaluated.',
            'Full adapters, training-feature parity and catalog publication remain outstanding.'],
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/validate_nasa_first_configuration_features.py','sciona/fixed_window_statistics.py','sciona/asof_indices.py','tests/test_asof_indices.py','sciona/configuration_state_features.py','tests/test_configuration_state_features.py','sciona/event_count_windows.py','tests/test_event_count_windows.py']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_configuration_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','future_configuration_feature_changes','event_count_comparisons','tied_timestamp_count_mismatches','approved']}))

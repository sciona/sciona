"""Synthetic source checks for rolling event-delay features and availability limits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.fixed_window_statistics import fixed_window_statistics
from sciona.event_delay_windows import event_delay_windows

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned source differs')
    function=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='extract_standtime_features')
    time=ast.literal_eval(function.body[0].value.func.value.slice)[0]
    mapping=next(n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='map')
    group=mapping.func.value.slice.value
    estimate=mapping.args[0].func.value.attr
    comparison=next(n for n in ast.walk(function) if isinstance(n,ast.Compare))
    actual=comparison.left.slice.value
    namespace={'pd':pd,'np':np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_standtime_features','exec'),namespace)
    run=namespace[function.name]
    origin=pd.Timestamp('2001-01-01')
    offsets=np.array([-1500,-1440,-721,-720,-361,-360,-241,-240,-121,-120,-61,-60,-1,0])
    observation=pd.DatetimeIndex([origin+pd.Timedelta(minutes=int(v)) for v in offsets])
    events=observation-pd.Timedelta(minutes=10)
    delays=np.arange(len(offsets))*3.-15
    identities=['synthetic_'+str(i) for i in range(len(offsets))]
    history=pd.DataFrame({group:identities,time:observation,actual:events})
    estimates=pd.DataFrame({group:identities,time:events-pd.Timedelta(minutes=30),estimate:events-pd.to_timedelta(delays,unit='min')})
    query_times=[origin-pd.Timedelta(days=3),origin,origin+pd.Timedelta(minutes=15)]
    queries=pd.DataFrame({time:query_times+[origin]})
    result=run(queries.copy(),history.copy(),estimates.copy())
    if len(result)!=3:raise ValueError('Duplicate query-time projection differs')
    rounded=observation.ceil('15min');checked=0
    kernel=fixed_window_statistics(rounded.as_unit('ns').asi8,observation.as_unit('ns').asi8,
        pd.DatetimeIndex(estimates[time]).as_unit('ns').asi8,delays.astype(np.float64),
        np.array([q.value for q in query_times],dtype=np.int64),
        np.array([pd.Timedelta(hours=h).value for h in [1,2,4,6,12,24]],dtype=np.int64))
    adapter=event_delay_windows(history[group].to_numpy(dtype=str),observation.as_unit('ns').asi8,
        events.as_unit('ns').asi8,estimates[group].to_numpy(dtype=str),pd.DatetimeIndex(estimates[time]).as_unit('ns').asi8,
        pd.DatetimeIndex(estimates[estimate]).as_unit('ns').asi8,estimates[estimate].notna().to_numpy(),
        np.array([q.value for q in query_times],dtype=np.int64),
        np.array([pd.Timedelta(hours=h).value for h in [1,2,4,6,12,24]],dtype=np.int64),
        int(pd.Timedelta(minutes=15).value),int(pd.Timedelta(minutes=1).value))
    for qi,query in enumerate(query_times):
        row=result.loc[result[time]==query].iloc[0]
        for wi,hours in enumerate([1,2,4,6,12,24]):
            included=(rounded>query-pd.Timedelta(hours=hours))&(rounded<=query)
            values=delays[included]
            for statistic,reference in [('maxdelay',np.max(values) if len(values) else np.nan),('meandelay',np.mean(values) if len(values) else np.nan),('count',float(len(values)))]:
                column=next(c for c in result if c.endswith('_'+statistic+'_'+str(hours)))
                np.testing.assert_allclose(row[column],reference,rtol=1e-14,atol=1e-14,equal_nan=True)
                position={'count':0,'meandelay':2,'maxdelay':3}[statistic]
                np.testing.assert_allclose(kernel[position][qi,wi],reference,rtol=1e-14,atol=1e-14,equal_nan=True)
                np.testing.assert_allclose(adapter[position][qi,wi],reference,rtol=1e-14,atol=1e-14,equal_nan=True)
                checked+=1
    # An unrestricted caller history can include an estimate unavailable at query time.
    # The source consumes it without an as-of condition; this establishes a needed
    # input availability contract, not a claim about actual competition records.
    one_history=pd.DataFrame({group:['synthetic_one'],time:[origin-pd.Timedelta(minutes=15)],actual:[origin-pd.Timedelta(minutes=20)]})
    future_estimate=pd.DataFrame({group:['synthetic_one'],time:[origin+pd.Timedelta(minutes=1)],estimate:[origin-pd.Timedelta(minutes=35)]})
    empty_estimate=future_estimate.iloc[:0].copy()
    one_query=pd.DataFrame({time:[origin]})
    with_future=run(one_query.copy(),one_history.copy(),future_estimate.copy())
    without_future=run(one_query.copy(),one_history.copy(),empty_estimate.copy())
    delay_columns=[c for c in with_future if 'maxdelay' in c or 'meandelay' in c]
    changed=sum(not np.isclose(with_future[c].iloc[0],without_future[c].iloc[0],equal_nan=True) for c in delay_columns)
    if changed!=12:raise ValueError('Availability counterexample differs')
    corrected=fixed_window_statistics(np.array([-15],dtype=np.int64),np.array([-15],dtype=np.int64),
        np.array([1],dtype=np.int64),np.array([15.]),np.array([0],dtype=np.int64),np.array([60,120,240,360,720,1440],dtype=np.int64))
    np.testing.assert_array_equal(corrected[0],np.ones((1,6),dtype=np.int64))
    np.testing.assert_array_equal(corrected[1],np.zeros((1,6),dtype=np.int64))
    if not np.isnan(corrected[2]).all() or not np.isnan(corrected[3]).all():raise ValueError('Future value leaked into corrected statistics')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        kernel_source_comparisons=checked,event_delay_adapter_source_comparisons=checked,corrected_future_value_outputs=12,
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            "sciona/fixed_window_statistics.py","tests/test_fixed_window_statistics.py","sciona/event_delay_windows.py","tests/test_event_delay_windows.py"]},synthetic_events=14,unique_queries=3,rolling_statistics_checked=checked,availability_counterexample_outputs=changed,
        findings=['Six time windows match independent right-closed, left-open references after upward 15-minute observation rounding.',
            'Duplicate query times collapse to one feature row; later keyed query joins must preserve requested row identities.',
            'The first estimate lookup has no query-time availability filter. An unrestricted future-observed estimate changes twelve synthetic delay aggregates.',
            'Event count is based on nonmissing identifiers, while delay aggregates ignore missing estimates; these populations can differ.'],
        required_contract='Caller must establish observation and actual-event availability for each query, or a corrected as-of implementation must filter explicitly and be requalified.',
        limitations=['Synthetic software behavior, not a claim of leakage in the original competition inputs or scorer.',
            'Corrected numerical event-delay adapter matches qualified cases; full raw-feature pipeline and publication approval remain outstanding.'],
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_standtime_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','rolling_statistics_checked','availability_counterexample_outputs','approved']}))

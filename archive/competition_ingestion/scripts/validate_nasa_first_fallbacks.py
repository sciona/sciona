"""Exercise pinned submission fallback control flow on synthetic inputs only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import typing
import numpy as np
import pandas as pd
from sciona.time_estimate_fallback import grouped_time_estimate_fallback
from sciona.asof_estimate_fallback import asof_estimate_fallback
from sciona.prediction_fallback_chain import prediction_fallback_chain

ROOT=Path(__file__).resolve().parents[1]
PINS={'1st Place/Phase 1/submission/solution.py':'854eb1f0874fa77748b26673cc848c8fe4dde38c76e7b474694069c05741e917',
      '1st Place/Phase 1/submission/utilities.py':'8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'}


def validate(source):
    functions={}
    for path,digest in PINS.items():
        data=(source/path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Pinned software differs')
        tree=ast.parse(data)
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in ['predict','baseline_predictions']:functions[n.name]=n
    baseline=functions['baseline_predictions']
    group_column=next(n.args[0].value for n in ast.walk(baseline) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='groupby')
    time_column=next(n.args[0].value for n in ast.walk(baseline) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sort_values')
    estimate_column=baseline.body[0].value.attr
    result_column=next(n.targets[0].slice.value for n in baseline.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript))
    class Logger:
        def info(self,*args):pass
    namespace={'pd':pd,'np':np,'Any':typing.Any,'Path':Path,'logger':Logger()}
    exec(compile(ast.Module(body=list(functions.values()),type_ignores=[]),'pinned_fallbacks','exec'),namespace)
    origin=pd.Timestamp('2001-01-01')
    query=pd.DataFrame({group_column:np.arange(6),time_column:[origin]*6})
    history=pd.DataFrame({group_column:[0,1,1,1,2,4,5],time_column:[origin-pd.Timedelta(minutes=i) for i in [1,3,2,1,1,1,1]],
        estimate_column:[origin+pd.Timedelta(minutes=v) if v is not None else pd.NaT for v in [20,45,60,None,1000,None,75.5]]})
    expected=np.array([1.,30.,299.,15.,15.,45.5])
    original=query.copy(deep=True)
    result=namespace['baseline_predictions'](query,history)
    np.testing.assert_array_equal(result[result_column].to_numpy(),expected)
    pd.testing.assert_frame_equal(query,original)
    # Source merge resets index, while subtraction aligns to caller index.
    nonrange=query.copy();nonrange.index=np.arange(100,106)
    drift=namespace['baseline_predictions'](nonrange,history)[result_column].to_numpy()
    if np.array_equal(drift,expected):raise ValueError('Index counterexample disappeared')
    np.testing.assert_array_equal(drift,np.full(6,15.))
    corrected=grouped_time_estimate_fallback(nonrange,history,group_column=group_column,order_column=time_column,estimate_column=estimate_column,query_time_column=time_column,seconds_per_unit=60,offset=30,lower=1,upper=299,missing=15)
    np.testing.assert_array_equal(corrected,expected)
    entry=functions['predict']
    arguments={a.arg:pd.DataFrame() for a in entry.args.args}
    arguments.update(airport='synthetic',prediction_time=origin,partial_submission_format=query.copy(),
        model={'synthetic':{}},solution_directory=Path('.'),etd=history)
    routes=[]
    def success(*args):routes.append('primary');return pd.DataFrame({'synthetic_result':[123]})
    namespace['catboost_predictions']=success
    pd.testing.assert_frame_equal(namespace['predict'](**arguments),pd.DataFrame({'synthetic_result':[123]}))
    if routes!=['primary']:raise ValueError('Primary route differs')
    def failure(*args):raise RuntimeError('synthetic model failure')
    namespace['catboost_predictions']=failure
    fallback=namespace['predict'](**arguments)
    np.testing.assert_array_equal(fallback[result_column].to_numpy(),expected)
    missing_model=namespace['predict'](**{**arguments,'model':{}})
    np.testing.assert_array_equal(missing_model[result_column].to_numpy(),expected)
    namespace['baseline_predictions']=failure
    manual_input=query.copy()
    manual=namespace['predict'](**{**arguments,'partial_submission_format':manual_input})
    np.testing.assert_array_equal(manual[result_column].to_numpy(),np.full(6,30))
    if manual is not manual_input or result_column not in manual_input:raise ValueError('Source constant fallback mutation differs')
    bare_handlers=sum(isinstance(n,ast.ExceptHandler) and n.type is None for n in ast.walk(entry))
    if bare_handlers!=2:raise ValueError('Exception scope differs')
    def available_baseline(records):
        return asof_estimate_fallback(records[group_column].to_numpy(dtype=str),
            pd.DatetimeIndex(records[time_column]).as_unit('ns').asi8,
            pd.DatetimeIndex(records[estimate_column]).as_unit('ns').asi8,records[estimate_column].notna().to_numpy(),
            nonrange[group_column].to_numpy(dtype=str),pd.DatetimeIndex(nonrange[time_column]).as_unit('ns').asi8,
            ticks_per_unit=60_000_000_000,offset=30,lower=1,upper=299,missing=15)
    np.testing.assert_array_equal(available_baseline(history),expected)
    future=history.iloc[[0]].copy();future[time_column]=origin+pd.Timedelta(minutes=1)
    future[estimate_column]=origin+pd.Timedelta(days=1)
    np.testing.assert_array_equal(available_baseline(pd.concat([history,future],ignore_index=True)),expected)
    corrected_routes=[]
    for primary,secondary,route,values in [
        (lambda:np.full(6,123),failure,'primary',np.full(6,123)),
        (failure,lambda:available_baseline(history),'baseline',expected),
        (lambda:np.full(6,np.nan),lambda:available_baseline(history),'baseline',expected),
        (failure,failure,'constant',np.full(6,30))]:
        predictions,selected_route=prediction_fallback_chain(primary,secondary,6,30)
        if selected_route!=route:raise ValueError('Corrected fallback routing differs')
        np.testing.assert_array_equal(predictions,values);corrected_routes.append(route)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        baseline_rows=6,control_flow_cases=4,nondefault_index_counterexamples=int(np.count_nonzero(drift!=expected)),
        corrected_available_baseline_rows=6,corrected_future_estimate_isolation=True,corrected_control_flow_cases=len(corrected_routes),
        corrected_nondefault_index_rows=6,implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['sciona/time_estimate_fallback.py','tests/test_time_estimate_fallback.py','sciona/asof_estimate_fallback.py','tests/test_asof_estimate_fallback.py','sciona/prediction_fallback_chain.py','tests/test_prediction_fallback_chain.py']},source_sha256=PINS,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        findings=['Grouped last-value selection skips missing estimates; it is not the estimate from the latest row unconditionally.',
            'Baseline subtracts a fixed offset, clips to bounds, fills missing values and retains fractional predictions.',
            'Primary errors and absent model entries select the baseline; a second failure selects the constant fallback.',
            'Nondefault caller indices cause merge/subtraction misalignment and default predictions in the synthetic counterexample.',
            'The constant fallback mutates the caller table, and two bare exception handlers catch beyond ordinary exceptions.'],
        required_corrections=['Corrected baseline uses explicit positional alignment; retain this source divergence in the final review.',
            'Define caller-input ownership and exception handling in the reusable execution API; preserve numeric fallback rules without silently inheriting mutation or catching process-control exceptions.'],
        limitations=['Synthetic baseline execution and fault-injected outer routing; feature extraction and native model errors are not simulated end to end.',
            'Corrected availability-aware baseline and routing are checked; complete native workflow fallback integration, CDG registration and publication approval remain outstanding.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_fallbacks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','baseline_rows','control_flow_cases','nondefault_index_counterexamples','approved']}))

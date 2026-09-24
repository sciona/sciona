"""Execute pinned estimated-time features on synthetic entity boundaries."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.estimate_history_features import estimate_history_features

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned source differs')
    tree=ast.parse(data)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_etd_features')
    pair_columns=ast.literal_eval(function.body[0].value.func.value.slice)
    group,time=pair_columns
    fill=next(n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='ffill')
    estimate=fill.func.value.slice.value
    changes=[n.targets[0].slice.elts[1].value for n in ast.walk(function) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript)
        and isinstance(n.targets[0].value,ast.Attribute) and n.targets[0].value.attr=='loc']
    change=changes[0]
    namespace={'pd':pd,'np':np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_estimate_features','exec'),namespace)
    run=namespace['extract_etd_features']
    origin=pd.Timestamp('2001-01-01')
    queries=pd.DataFrame({group:['synthetic_a','synthetic_b'],time:[origin,origin]})
    history=pd.DataFrame({group:['synthetic_a','synthetic_a','synthetic_b','synthetic_b'],
        time:[origin+pd.Timedelta(minutes=x) for x in [-30,-15,-30,-15]],
        estimate:[origin+pd.Timedelta(minutes=x) for x in [60,70,180,150]]})
    result=run(queries.copy(),history.copy()).sort_values([group,time]).reset_index(drop=True)
    np.testing.assert_array_equal(result[change],[0,10,0,0,-30,0])
    perturbed=history.copy()
    perturbed.loc[perturbed[group]=='synthetic_a',estimate]+=pd.Timedelta(days=3)
    other=run(queries.copy(),perturbed).sort_values([group,time]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result[result[group]=='synthetic_b'].reset_index(drop=True),other[other[group]=='synthetic_b'].reset_index(drop=True),check_exact=True)
    # Future observations must not change any features at the earlier query.
    future=pd.DataFrame({group:['synthetic_a'],time:[origin+pd.Timedelta(minutes=1)],estimate:[origin+pd.Timedelta(days=9)]})
    extended=run(queries.copy(),pd.concat([history,future],ignore_index=True)).sort_values([group,time]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result[result[time]<=origin].reset_index(drop=True),extended[extended[time]<=origin].reset_index(drop=True),check_exact=True)
    shuffled=run(queries.copy(),history.iloc[[3,0,2,1]].copy()).sort_values([group,time]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result,shuffled,check_exact=True)
    # Source preserves history rows too; downstream query join is mandatory.
    if len(result)!=6 or len(queries)!=2:raise ValueError('Expected source history-inclusive output differs')
    selected=queries.merge(result,on=pair_columns,how='left',validate='one_to_one')
    if len(selected)!=2:raise ValueError('Query projection differs')
    windows=[60,120,240,360,720,1440]
    rolling=[c for c in result.columns if '_roll_' in c]
    if len(rolling)!=18:raise ValueError('Complete rolling coverage differs')
    for minutes in windows:
        for stat in ['sum','max','std']:
            column=next(c for c in rolling if c.endswith('_'+str(minutes)+'_'+stat))
            for identity,expected in [('synthetic_a',[0.,10.,0.]),('synthetic_b',[0.,-30.,0.])]:
                values=np.asarray(expected)
                reference=values.sum() if stat=='sum' else values.max() if stat=='max' else values.std(ddof=1)
                actual=selected.loc[selected[group]==identity,column].iloc[0]
                np.testing.assert_allclose(actual,reference,rtol=1e-14,atol=1e-14)
    # Observations are rounded upward to a 15-minute grid before joining.
    # A missing later estimate in the same bin does not erase an earlier value.
    offset_queries=pd.DataFrame({group:['synthetic_a']*2,time:[origin+pd.Timedelta(minutes=v) for v in [5,15]]})
    offset_history=pd.DataFrame({group:['synthetic_a']*2,time:[origin+pd.Timedelta(minutes=v) for v in [1,2]],
        estimate:[origin+pd.Timedelta(minutes=60),pd.NaT]})
    offset_result=run(offset_queries.copy(),offset_history.copy())
    offset_selected=offset_queries.merge(offset_result,on=pair_columns,how='left',validate='one_to_one')
    remaining=changes[-1]
    if not pd.isna(offset_selected[remaining].iloc[0]) or offset_selected[remaining].iloc[1]!=45:
        raise ValueError('Rounded observation availability or last-nonmissing behavior differs')
    extra_query=pd.DataFrame({group:['synthetic_a'],time:[origin-pd.Timedelta(minutes=5)]})
    expanded=run(pd.concat([queries,extra_query],ignore_index=True),history.copy())
    expanded_selected=queries.merge(expanded,on=pair_columns,how='left',validate='one_to_one')
    changed_statistics=sum(not np.isclose(selected.loc[selected[group]=='synthetic_a',c].iloc[0],
        expanded_selected.loc[expanded_selected[group]=='synthetic_a',c].iloc[0],equal_nan=True) for c in rolling)
    if changed_statistics!=6:raise ValueError('Expected query-population dependence not reproduced')
    generic_args=[history[group].to_numpy(),np.array([-30,-15,-30,-15],dtype=np.int64),
        np.array([60,70,180,150],dtype=np.int64),np.ones(4,dtype=bool),queries[group].to_numpy(),
        np.zeros(2,dtype=np.int64),np.array(windows,dtype=np.int64),15,1]
    corrected=estimate_history_features(*generic_args)
    corrected_comparisons=0
    for wi,width in enumerate(windows):
        for stat in ['sum','max','std']:
            column=next(c for c in rolling if c.endswith('_'+str(width)+'_'+stat))
            np.testing.assert_allclose(corrected[stat][:,wi],selected[column].to_numpy(),rtol=1e-14,atol=1e-14)
            corrected_comparisons+=len(selected)
    np.testing.assert_array_equal(corrected['remaining'],selected[remaining].to_numpy())
    generic_args[4]=np.array(['synthetic_a','synthetic_b','synthetic_a'])
    generic_args[5]=np.array([0,0,-5],dtype=np.int64)
    corrected_expanded=estimate_history_features(*generic_args)
    for key in corrected:
        np.testing.assert_array_equal(corrected[key],corrected_expanded[key][:2])
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        synthetic_entities=2,source_output_rows=6,query_output_rows=2,rolling_statistics_checked=36,rounded_availability_queries=2,last_nonmissing_bin_verified=True,
        boundary_reset_verified=True,unrelated_entity_perturbation_invariant=True,future_observation_invariant=True,distinct_time_shuffle_invariant=True,
        extra_query_changed_rolling_statistics=changed_statistics,
        corrected_rolling_source_comparisons=corrected_comparisons,corrected_query_population_invariance=True,
        findings=['Ungrouped shifts are followed by an explicit entity-boundary reset. The tested entity boundary does not leak changes across entities.',
            'The source returns history rows as well as query rows; the later keyed query join is required.',
            'Six time windows produce sum, maximum and sample-standard-deviation change features; all query statistics match independent references.',
            'Adding an earlier query for the same entity inserts a zero change into each window and changes six standard-deviation features at an existing query.'],
        limitations=['Synthetic two-entity execution, distinct observation times plus one off-grid/null bin; broader duplicate histories, exact window boundaries and timezone regimes remain to be checked.',
            'The corrected component evaluates history bins plus each current query independently; it changes source batch-dependent statistics and requires retraining with the same policy.',
            'Full raw-field adaptation and catalog publication remain outstanding.'],
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['sciona/estimate_history_features.py','tests/test_estimate_history_features.py']},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_estimate_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,boundary_reset_verified=True,rolling_statistics_checked=36,approved=False)))

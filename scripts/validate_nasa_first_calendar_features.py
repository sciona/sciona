"""Compare calendar arithmetic with pinned source using an explicit synthetic schedule."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.calendar_time_features import calendar_time_features

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned source differs')
    function=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='extract_moment_features')
    time=ast.literal_eval(function.body[0].value.func.value.slice)[0]
    fields=[n.targets[0].slice.value for n in function.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript)]
    if len(fields)!=8:raise ValueError('Calendar feature arity differs')
    holidays=pd.date_range('1999-12-31','2001-02-01',freq='9D')
    class SyntheticCalendar:
        def holidays(self,**kwargs):return holidays.copy()
    namespace={'pd':pd,'np':np,'calendar':SyntheticCalendar,'START_HOLIDAY':None,'END_HOLIDAY':None}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_calendar','exec'),namespace)
    rng=np.random.default_rng(529)
    queries=pd.DatetimeIndex(pd.Timestamp('2000-01-01')+pd.to_timedelta(rng.integers(0,366*24*60,size=2048),unit='min'))
    queries=queries.append(pd.DatetimeIndex([holidays[3],holidays[3]+pd.Timedelta(hours=12),holidays[3],pd.Timestamp('2001-01-01')]))
    unique=queries.drop_duplicates()
    expected=namespace[function.name](pd.DataFrame({time:queries}))
    actual=calendar_time_features(unique,holidays)
    keys=['hour','weekday','month','day','iso_week','week_of_month','is_holiday','days_until_holiday']
    for name,key in zip(fields,keys):np.testing.assert_array_equal(expected[name].to_numpy(),actual[key])
    beyond=pd.DatetimeIndex([holidays[-1]+pd.Timedelta(hours=1)])
    try:namespace[function.name](pd.DataFrame({time:beyond}))
    except ValueError as error:
        if str(error) not in ['min() iterable argument is empty','min() arg is an empty sequence']:raise
    else:raise ValueError('Expected source horizon failure not reproduced')
    try:calendar_time_features(beyond,holidays)
    except ValueError as error:
        if 'cover query horizon' not in str(error):raise
    else:raise ValueError('Missing explicit horizon rejection')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        unique_queries=len(unique),features=8,exact_feature_values=len(unique)*8,source_horizon_failure_reproduced=True,
        explicit_horizon_rejection=True,
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['sciona/calendar_time_features.py','tests/test_calendar_time_features.py','scripts/validate_nasa_first_calendar_features.py']},
        semantics=['Holiday membership uses the calendar date; countdown compares exact timestamp to holiday midnights and floors elapsed days.',
            'Timezone-naive queries and a sorted, unique explicit holiday schedule; duplicate source query timestamps are projected explicitly.',
            'Schedule exhaustion is rejected clearly instead of relying on min of an empty sequence.'],
        limitations=['Calendar provider and configured coverage are injected synthetic inputs; no specific regional holiday rules or original configuration qualification.',
            'No full raw-feature graph or publication approval applied.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_calendar_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','unique_queries','exact_feature_values','explicit_horizon_rejection','approved']}))

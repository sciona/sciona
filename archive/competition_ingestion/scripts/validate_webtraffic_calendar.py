"""Source calendar lookup and independent Gregorian lag checks on synthetic dates."""
import argparse
import ast
import calendar
import datetime
import hashlib
import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sciona.webtraffic_calendar import lag_indexes


def validate(root,source):
    path=source/'make_features.py'
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['make_features.py']
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='lag_indexes')
    class LegacySeries(pd.Series):
        @property
        def loc(self):
            class Lookup:
                def __getitem__(_,dates):return self.reindex(dates)
            return Lookup()
    class LegacyPandas:
        Series=LegacySeries
        def __getattr__(self,name):return getattr(pd,name)
    ns=dict(np=np,pd=LegacyPandas(),List=List)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<original-calendar>','exec'),ns)
    # Ordinary current pandas rejects missing labels in the original .loc call.
    raw=dict(np=np,pd=pd,List=List)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<unadapted-calendar>','exec'),raw)
    try:raw['lag_indexes']('2000-01-01','2000-01-03')
    except KeyError:pass
    else:raise AssertionError('Expected legacy lookup incompatibility')
    counts=dict(date_ranges=0,lag_arrays=0,available=0,unavailable=0)
    for start in ['1899-12-01','1999-12-01','2004-02-29','2023-01-31']:
        for days in [1,95,400,800]:
            first=datetime.date.fromisoformat(start);last=first+datetime.timedelta(days=days-1)
            actual=lag_indexes(first,last);expected=ns['lag_indexes'](first,last)
            dates=[first+datetime.timedelta(days=i) for i in range(days)]
            lookup={d:i for i,d in enumerate(dates)}
            for months,a,b in zip([3,6,9,12],actual,expected):
                pd.testing.assert_series_equal(a,pd.Series(b))
                indices=[]
                for date in dates:
                    month_index=date.year*12+date.month-1-months
                    year,month=divmod(month_index,12);month+=1
                    day=min(date.day,calendar.monthrange(year,month)[1])
                    indices.append(lookup.get(datetime.date(year,month,day),-1))
                np.testing.assert_array_equal(a.to_numpy(),np.array(indices,dtype=np.int16))
                assert a.dtype==np.int16
                counts['available']+=int((a>=0).sum());counts['unavailable']+=int((a==-1).sum())
                counts['lag_arrays']+=1
            counts['date_ranges']+=1
    paths=['sciona/webtraffic_calendar.py','scripts/validate_webtraffic_calendar.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Legacy .loc missing-date behavior explicitly adapted to reindex; raw current-pandas failure reproduced.',
                     'Naive daily ranges up to 800 days tested; source int16 storage retained, no long-range overflow generalization.',
                     'Calendar lags only; page features and full training/inference remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_calendar.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

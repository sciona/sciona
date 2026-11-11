"""Synthetic row/key alignment and copy-boundary checks for query routing."""
import hashlib
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sciona.atoms.ml.domain_adapters import osic_query_routing as provider

ROOT=Path(__file__).resolve().parents[1]


def validate():
    original=pd.DataFrame(dict(key=[' z ', 'a', 'm', 'b'],
        entity=['second', 'first', 'second', 'second'], time=[2, -1, 2, 8]),
        index=[7, 3, 7, 0])
    expected=original.loc[:, ['entity', 'time']].copy(deep=True)
    cases=0
    for columns in itertools.permutations(original.columns):
        frame=original.loc[:, list(columns)].copy(deep=True)
        before=frame.copy(deep=True)
        rows,keys=provider.split_keyed_queries(frame)
        pd.testing.assert_frame_equal(rows, expected)
        assert keys==[' z ', 'a', 'm', 'b'] and type(keys) is list
        assert not np.shares_memory(rows['time'].to_numpy(), frame['time'].to_numpy())
        rows.iloc[0, rows.columns.get_loc('time')]=999
        rows.iloc[0, rows.columns.get_loc('entity')]='changed'
        keys[0]='changed'
        pd.testing.assert_frame_equal(frame,before)
        cases+=1
    extremes=original.copy(deep=True)
    extremes['time']=np.array([np.iinfo(np.int64).min, 0, 1, np.iinfo(np.int64).max],dtype=np.int64)
    rows,_=provider.split_keyed_queries(extremes)
    np.testing.assert_array_equal(rows['time'].to_numpy(),extremes['time'].to_numpy())
    class FrameSubclass(pd.DataFrame):pass
    invalid=[('not_frame',{}),('subclass',FrameSubclass(original)),('empty',original.iloc[:0]),
        ('missing_key',original.drop(columns='key')),('extra_column',original.assign(extra=0)),
        ('duplicate_columns',pd.concat([original,original[['key']]],axis=1)),
        ('duplicate_key',original.assign(key=['x']*4)),
        ('float_time',original.assign(time=original.time.astype('float64'))),
        ('int32_time',original.assign(time=original.time.astype('int32'))),
        ('nullable_time',original.assign(time=pd.array([1,2,3,4],dtype='Int64'))),
        ('boolean_time',original.assign(time=True))]
    for field in ('key','entity'):
        for label,value in [('blank',' '),('null',None),('nonstring',1)]:
            bad=original.copy(deep=True)
            bad.iloc[0,bad.columns.get_loc(field)]=value
            invalid.append((field+'_'+label,bad))
    rejected=[]
    for label,frame in invalid:
        try:provider.split_keyed_queries(frame)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Invalid queries accepted: '+label)
    try:provider.witness_queries()
    except ValueError:rejected.append('abstract_execution')
    else:raise AssertionError('Abstract routing accepted')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        column_orders_checked=cases,row_and_key_order_exact=True,duplicate_index_preserved=True,
        repeated_coordinates_preserved=True,exact_strings_preserved=True,
        output_mutations_leave_input_unchanged=True,int64_extremes_preserved=True,
        rejected_contracts=rejected,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Source-schema boundary only; no empirical effectiveness or whole-CDG approval.',
            'Supplied keys are opaque. Caller owns semantic identity, coordinate units and exact frozen query-scope agreement.',
            'No key generation, sorting, merging, fitting or numerical conversion.',
            'Only independent cell-value mutation is qualified; pandas index internals and concurrent mutation are outside this contract.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/osic_query_routing.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,column_orders=result['column_orders_checked'],rejected=len(result['rejected_contracts']))))

"""Candidate state and greedy commitment parity using synthetic tracks only."""
import argparse
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sciona.trackml_candidates import Candidates, zeroUsedHits


def greedy(matrix, used, minimum, maximum_loss, fraction, maximum_rows, reserve):
    """Sequential list/set oracle; synthetic rows contain no repeated nonzero IDs."""
    occupied=set(np.flatnonzero(used));occupied.add(0)
    result=np.zeros_like(matrix);accepted=0
    for i,row in enumerate(matrix):
        present=[int(h) for h in row if h]
        fresh=[h for h in present if h not in occupied]
        lost=len(present)-len(fresh)
        take=(len(fresh)>=minimum and lost<=maximum_loss and lost/len(present)<=fraction
              and accepted<maximum_rows)
        if take:
            result[i]=[h if h in fresh else 0 for h in row]
            accepted+=1
        if take or reserve:occupied.update(present)
    flags=np.array([i in occupied for i in range(len(used))])
    return result,flags


def validate(root,source):
    path=source/'trackml_solution/candidates.py'
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/candidates.py']
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    ns=dict(np=np,pd=pd,OrderedDict=OrderedDict)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-candidates>','exec'),ns)
    counts=dict(updates=0,fit_accesses=0,greedy_cases=0,submissions=0)
    event=SimpleNamespace(max_hit_id=1000,event_id=1,hits_df=pd.DataFrame({'hit_id':np.arange(1,1001)}))
    def same(a,b):
        for name in ['open','ncross','_candidates','_fit']:
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        pd.testing.assert_frame_equal(a.df,b.df)
    for seed in range(8):
        rng=np.random.RandomState(1431+seed)
        matrix=np.array([rng.choice(np.arange(1,65),6,replace=False) for _ in range(16)],dtype=np.int32)
        frame=pd.DataFrame({'rank':np.arange(16),'score':rng.uniform(size=16)})
        a=Candidates(event,nmax_per_crossing=2);b=ns['Candidates'](event,nmax_per_crossing=2)
        a.initialize(matrix.copy(),frame.copy());b.initialize(matrix.copy(),frame.copy());same(a,b)
        for step in range(6):
            vals=rng.normal(size=a.n);mask=rng.choice([False,True],a.n)
            for crossing in [0,-1,-99]:
                a.setFit(crossing,'hel_r',vals[mask],mask=mask)
                b.setFit(crossing,'hel_r',vals[mask],mask=mask)
                np.testing.assert_array_equal(a.getFit(crossing,'hel_r'),b.getFit(crossing,'hel_r'))
                counts['fit_accesses']+=1
            close=rng.choice([False,True],a.n);keep=rng.choice([False,True],a.n)
            extend=rng.choice(a.n,8,replace=True)
            ids=[np.arange(200+step*16,208+step*16,dtype=np.int32),
                 np.arange(208+step*16,216+step*16,dtype=np.int32)]
            previous=a._candidates.copy();previous_cross=a.ncross.copy();previous_df=a.df.copy()
            a.update(close_mask=close,keep_mask=keep,extend_index=extend,extend_hit_ids=ids)
            b.update(close_mask=close,keep_mask=keep,extend_index=extend,extend_hit_ids=ids)
            same(a,b);counts['updates']+=1
            backrefs=list(np.flatnonzero(keep))+list(extend)
            for row,parent in enumerate(backrefs):
                expected=np.zeros(a._candidates.shape[1],dtype=np.int32)
                expected[:previous.shape[1]]=previous[parent]
                if row>=keep.sum():
                    j=row-int(keep.sum());start=2*int(previous_cross[parent])
                    expected[start:start+2]=[ids[0][j],ids[1][j]]
                np.testing.assert_array_equal(a._candidates[row],expected)
            pd.testing.assert_frame_equal(a.df,previous_df.iloc[backrefs].reset_index(drop=True))
            copy=a.copy();copy._candidates[0,0]=999
            assert a._candidates[0,0]!=999
            order=rng.permutation(a.n)
            a.permute(order);b.permute(order);same(a,b)
        a.update(keep_mask=np.arange(a.n)%2==0);b.update(keep_mask=np.arange(b.n)%2==0);same(a,b)
        counts['updates']+=1
        # Greedy overlap resolution tests use the original overlapping seed rows.
        for minimum in [1,3,6]:
            for reserve in [False,True]:
                for maxrows in [2,16]:
                    used=np.zeros(1001,dtype=bool);used[[2,5,11]]=True
                    options=dict(min_nhits=minimum,max_nloss=2,max_loss_fraction=.5,
                                 max_nrows=maxrows,reserve_skipped=reserve)
                    am=matrix.copy();bm=matrix.copy();au=used.copy();bu=used.copy()
                    zeroUsedHits(am,au,**options);ns['zeroUsedHits'](bm,bu,**options)
                    oracle,ou=greedy(matrix,used,minimum,2,.5,maxrows,reserve)
                    for x,y in [(am,bm),(au,bu),(am,oracle),(au,ou)]:np.testing.assert_array_equal(x,y)
                    counts['greedy_cases']+=1
                    ac=Candidates(event,nmax_per_crossing=2);bc=ns['Candidates'](event,nmax_per_crossing=2)
                    ac.initialize(matrix.copy(),frame.copy());bc.initialize(matrix.copy(),frame.copy())
                    for fill in [False,True]:
                        actual=ac.submit(fill=fill,used=used,min_track_id=7,**options)
                        expected=bc.submit(fill=fill,used=used,min_track_id=7,**options)
                        pd.testing.assert_frame_equal(actual,expected)
                        assert actual.hit_id.is_unique
                        for row in actual.itertuples():
                            if row.track_id:
                                assert row.hit_id in oracle[row.track_id-7]
                            else:assert not ou[row.hit_id]
                        np.testing.assert_array_equal(ac._candidates,matrix)
                        assert not used[0]  # Submission copies its mutable used mask.
                        counts['submissions']+=1
    paths=['sciona/trackml_candidates.py','scripts/validate_trackml_candidates.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Synthetic rows have no repeated nonzero hit within a row; overlap across rows is exercised.',
                     'Candidate capacity limits, complete extension integration and full tracking lifecycle remain.',
                     'Submission here is an in-memory DataFrame; no external upload or catalog approval.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

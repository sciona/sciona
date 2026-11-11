"""Pinned numerical preprocessing parity on synthetic series, without TensorFlow."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sciona import webtraffic_features as candidate


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    path=source/'make_features.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['make_features.py']
    class LegacyNumpy:
        NaN=np.nan
        def __getattr__(self,name):return getattr(np,name)
    ns=dict(np=LegacyNumpy(),pd=pd,Tuple=Tuple)
    names={'single_autocorr','batch_autocorr','find_start_end','prepare_data'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in nodes:n.decorator_list=[]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-webtraffic-features>','exec'),ns)
    counts=dict(boundaries=0,preparation=0,autocorrelation=0)
    for seed in range(8):
        for dtype in [np.float32,np.float64]:
            rng=np.random.RandomState(2131+seed)
            data=rng.poisson(3,size=(16,800)).astype(dtype)
            data[rng.uniform(size=data.shape)<.1]=np.nan
            data[0]=0;data[1]=np.nan;data[2]=0;data[2,400]=1;data[3]=2
            data[4,:100]=0;data[4,700:]=0
            before=data.copy()
            a=candidate.find_start_end(data);b=ns['find_start_end'](data)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            for i,row in enumerate(data):
                positions=np.flatnonzero(row>0)
                assert a[0][i]==(positions[0] if len(positions) else -1)
                assert a[1][i]==(positions[-1] if len(positions) else -1)
            counts['boundaries']+=1
            frame=pd.DataFrame(data)
            ns['read_x']=lambda start,end:frame.copy(deep=True)
            for threshold in [0.,.5,799/800,1.]:
                actual=candidate.prepare_series(frame,threshold)
                with redirect_stdout(io.StringIO()):expected=ns['prepare_data'](None,None,threshold)
                for x,y in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(x,y)
                for x,y in zip(actual[2:],expected[2:]):np.testing.assert_array_equal(x,y)
                keep=(a[1]-a[0])/800>=threshold
                assert len(actual[0])==int(keep.sum())
                np.testing.assert_array_equal(actual[1],np.isnan(data[keep]))
                counts['preparation']+=1
            # Correlation receives source-prepared zero-filled log values.
            processed=candidate.prepare_series(frame,0.)[0].to_numpy()
            for lag in [7,30,365]:
                for backoffset in [0,50]:
                    with np.errstate(invalid='ignore',divide='ignore'):
                        actual=candidate.batch_autocorr(processed,lag,*a,threshold=1.5,backoffset=backoffset)
                        expected=ns['batch_autocorr'](processed,lag,*a,threshold=1.5,backoffset=backoffset)
                    np.testing.assert_array_equal(actual,expected)
                    for i,row in enumerate(processed):
                        end=min(a[1][i],800-backoffset)
                        if (end-a[0][i])/lag<=1.5:assert np.isnan(actual[i]);continue
                        segment=row[a[0][i]:end]
                        if np.ptp(segment)==0:continue  # Source float32 mean-rounding artifact; exact parity checked above.
                        parts=[]
                        for k in [lag,lag-1,lag+1]:
                            left=segment[k:];right=segment[:-k]
                            if np.std(left)==0 or np.std(right)==0:parts.append(0.)
                            else:parts.append(np.corrcoef(left,right)[0,1])
                        np.testing.assert_allclose(actual[i],np.dot(parts,[.5,.25,.25]),rtol=1e-5,atol=1e-7)
                    counts['autocorrelation']+=1
            np.testing.assert_array_equal(data,before)
    paths=['sciona/webtraffic_features.py','scripts/validate_webtraffic_features.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Original function bodies without numba decorators; no compiled-numba parity claim.',
                     'Source inclusive end index is used as exclusive correlation slice and validity length excludes one endpoint; preserved.',
                     'Constant float32 series may yield source correlation 1 from rounded means; independent correlation check excludes constants.',
                     'Feature subset only; calendar/page features, neural training and final ensemble remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

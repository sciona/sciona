"""Pinned-source greedy histogram comparison on synthetic populations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.amex_binning import boundaries,normalize
SOURCE_SHA='aca38743b8264103736089292abca7962176bc61d848c7c1af6bb2f2b8084377'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Source pin differs')
    funcs=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='GreedyFindBin'];assert len(funcs)==1
    ns={'__builtins__':{'list':list,'float':float,'range':range,'min':min,'max':max}}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<pinned bin source>','exec'),ns)
    rng=np.random.default_rng(107);cases=0
    for n in (1,2,8,50,256,300):
        for budget in (1,2,3,16,255):
            for _ in range(10):
                v=np.arange(n,dtype=float);c=rng.integers(1,50,n)
                expected=ns['GreedyFindBin'](v,c,n,budget,c.sum())
                np.testing.assert_array_equal(boundaries(v,c,max_bins=budget),expected);cases+=1
    x=rng.integers(0,300,(1000,3)).astype(float);x[0,:]=np.nan
    expected=np.empty_like(x)
    for j in range(3):
        v,c=np.unique(x[~np.isnan(x[:,j]),j],return_counts=True)
        bins=ns['GreedyFindBin'](v,c,len(v),255,c.sum())
        code=np.digitize(x[:,j],[-np.inf]+bins);code[code==len(bins)+1]=0
        expected[:,j]=code/code.max()
    np.testing.assert_array_equal(normalize(x)['values'],expected)
    files=[ROOT/'sciona/amex_binning.py',ROOT/'tests/test_amex_binning.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,source_software_sha256=SOURCE_SHA,histogram_cases=cases,normalized_columns=3,
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Independent numerical reconstruction with source compressed split omission retained. All-missing histograms explicitly reject source undefined input.',
            'Synthetic feature transforms only; model execution and publication pending.'])
    (ROOT/'docs/reviews/competition_amex_binning_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='passed',histogram_cases=cases,normalized_columns=3)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True);main(p.parse_args().source_code)

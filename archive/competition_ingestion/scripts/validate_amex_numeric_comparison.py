"""Compare numeric summaries with pinned private source using synthetic sequences."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.amex_numeric import summarize
SOURCE_SHA='6de0a4d65a17a0a25e10c8fdf19fa8efca6babd94aef1f6aabf71a72334a8911'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Pinned source differs')
    functions=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('num_feature','diff_feature')]
    assert len(functions)==2
    ns={'np':np,'pd':pd,'__builtins__':{'print':lambda *a:None}}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned numeric source>','exec'),ns)
    rng=np.random.default_rng(102);cases=0
    for size in (1,2,7,13):
        x=rng.integers(-20,20,size=(size,2)).astype(float);x[-1,0]=np.nan
        for last in (None,3):
            for differences,ranked in ((False,False),(True,False),(False,True)):
                names=['rank_a','rank_b'] if ranked else ['a','b'];ns.update(num_features=names,lastk=last)
                selected=x if last is None else x[-last:]
                df=pd.DataFrame(selected,columns=names);df.insert(0,'customer_ID',np.zeros(len(df),int))
                expected=ns['diff_feature' if differences else 'num_feature'](df).iloc[0,1:].to_numpy(dtype=float)
                np.testing.assert_allclose(summarize(x,last_window=last,differences=differences,ranked=ranked),expected,atol=0,rtol=0,equal_nan=True)
                cases+=1
    files=[ROOT/'sciona/amex_numeric.py',ROOT/'tests/test_amex_numeric.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,source_software_sha256=SOURCE_SHA,comparison_cases=cases,
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Independent summaries compared on synthetic numeric sequences only; floating-point aggregation order may differ outside these cases.',
            'Denoising categorical mappings, full manual/rank features, sequence/binning paths, learners, blend and graph remain pending. Original source reuse terms unresolved; no source code copied into repository.'])
    (ROOT/'docs/reviews/competition_amex_numeric_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',comparison_cases=cases)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True);main(p.parse_args().source_code)

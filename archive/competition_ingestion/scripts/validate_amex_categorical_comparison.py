"""Pinned-source categorical comparison using only synthetic groups and codes."""
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
from sciona.amex_categorical import summarize_populations
SOURCE_SHA='6de0a4d65a17a0a25e10c8fdf19fa8efca6babd94aef1f6aabf71a72334a8911'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Pinned source differs')
    functions=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('one_hot_encoding','cat_feature')]
    assert len(functions)==2
    ns={'np':np,'pd':pd,'__builtins__':{'print':lambda *a:None}};exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned categorical source>','exec'),ns)
    rng=np.random.default_rng(103);cases=0
    for widths in (1,3):
        xs=[rng.integers(0,4,(n,widths)).astype(float) for n in (1,3,6,13)]
        for x in xs:x[-1,0]=np.nan
        masks=[np.arange(len(x))%2==0 for x in xs]
        for windowed in (False,True):
            names=[f'c{i}' for i in range(widths)];ns.update(cat_features=names,lastk=3 if windowed else None)
            df=pd.DataFrame(np.vstack(xs),columns=names)
            df['customer_ID']=np.repeat(np.arange(len(xs)),[len(x) for x in xs])
            df['S_2']=np.where(np.concatenate(masks),1.,np.nan)
            encoded=ns['one_hot_encoding'](df,names,False)
            expected=ns['cat_feature'](encoded).iloc[:,1:].to_numpy(dtype=float)
            actual=summarize_populations(xs,masks,windowed=windowed)['values']
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-13,equal_nan=True);cases+=1
    files=[ROOT/'sciona/amex_categorical.py',ROOT/'tests/test_amex_categorical.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,comparison_cases=cases,source_software_sha256=SOURCE_SHA,
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Joint numeric-category vocabulary and summaries only; source string mappings and rank/window orchestration remain pending.',
            'Synthetic source comparison, not complete trainer or publication qualification.'])
    (ROOT/'docs/reviews/competition_amex_categorical_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',comparison_cases=cases)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True);main(p.parse_args().source_code)

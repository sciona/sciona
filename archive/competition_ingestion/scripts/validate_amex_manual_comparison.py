"""Compare all independent manual blocks with pinned source numerical helpers."""
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
from sciona.amex_manual import build
SOURCE_SHA='6de0a4d65a17a0a25e10c8fdf19fa8efca6babd94aef1f6aabf71a72334a8911'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Source pin differs')
    wanted={'one_hot_encoding','cat_feature','num_feature','diff_feature'}
    funcs=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert len(funcs)==4
    ns={'np':np,'pd':pd,'__builtins__':{'print':lambda *a:None}}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<pinned manual source>','exec'),ns)
    rng=np.random.default_rng(105);lengths=[2,7,13]
    numeric=[rng.integers(-10,10,(n,2)).astype(float) for n in lengths]
    categorical=[rng.integers(0,3,(n,2)).astype(float) for n in lengths]
    for x in numeric:x[-1,0]=np.nan
    times=[rng.permutation(n).astype(float) for n in lengths];months=[np.arange(n)%3 for n in lengths]
    actual=build(numeric,categorical,times,months,zero_fill_columns=[0]);expected={}
    df=pd.DataFrame(np.vstack(numeric),columns=['a','b']);df['a']=df['a'].fillna(0)
    df[['c','d']]=np.vstack(categorical);df['customer_ID']=np.repeat(np.arange(3),lengths);df['S_2']=np.concatenate(times);df['month']=np.concatenate(months)
    def values(frame):return frame.set_index('customer_ID').reindex(range(3)).to_numpy(dtype=float)
    for window in (None,3,6):
        part=df.copy()
        if window is not None:part=part.loc[part.groupby('customer_ID')['S_2'].rank(ascending=False)<=window].copy()
        ns.update(lastk=window,num_features=['a','b'],cat_features=['c','d'])
        prefix='full' if window is None else f'last{window}'
        expected[prefix+'_numeric']=values(ns['num_feature'](part))
        if window!=6:
            expected[prefix+'_difference']=values(ns['diff_feature'](part))
            expected[prefix+'_categorical']=values(ns['cat_feature'](ns['one_hot_encoding'](part,['c','d'],False)))
    for key,group,prefix in [('customer_rank','customer_ID','rank_'),('month_rank','month','ym_rank_')]:
        ranks=df.groupby(group)[['a','b']].rank(pct=True).add_prefix(prefix);ranks.insert(0,'customer_ID',df['customer_ID'])
        ns.update(lastk=None,num_features=[prefix+'a',prefix+'b'])
        expected[key]=values(ns['num_feature'](ranks))
    assert set(actual)==set(expected)
    for key in expected:
        tolerance=1e-14 if key.endswith('categorical') else 0
        np.testing.assert_allclose(actual[key],expected[key],atol=tolerance,rtol=0,equal_nan=True,err_msg=key)
    files=sorted((ROOT/'sciona').glob('amex_*.py'))+[ROOT/'tests/test_amex_manual.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,source_software_sha256=SOURCE_SHA,compared_blocks=9,
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Synthetic joint population comparison of manual blocks only. Inputs already denoised and category-mapped; final learner concatenation remains pending.',
            'Explicit runtime numeric/category/month/time/fill roles generalize private source schema; no source dataset contents included.'])
    (ROOT/'docs/reviews/competition_amex_manual_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='passed',compared_blocks=9)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True);main(p.parse_args().source_code)

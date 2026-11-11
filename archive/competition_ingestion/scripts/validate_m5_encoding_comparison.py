"""Compare grouped moments with a pinned numerical source loop on synthetic data."""
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
from sciona.m5_encoding import encode


def compare(source_root):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pins_path=ROOT/'docs/reviews/competition_m5_source_pins.json'
    pins=json.loads(pins_path.read_text())
    paths=list(source_root.rglob('1. preprocessing.py'))
    expected_hashes=[v for k,v in pins['files'].items() if k.endswith('1. preprocessing.py')]
    if len(paths)!=1 or len(expected_hashes)!=1 or sha(paths[0])!=expected_hashes[0]:
        raise ValueError('Unique pinned source required')
    tree=ast.parse(paths[0].read_text())
    inventories=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(x,ast.Name) and x.id=='icols' for x in n.targets)]
    loops=[n for n in tree.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='icols']
    icols=inventories[-1];loop=loops[-1]
    assert len(icols)==11
    loop.body=[n for n in loop.body if not isinstance(n,ast.Expr)]
    roles=list(dict.fromkeys(v for g in icols for v in g))
    target=next(n.slice.value for n in ast.walk(loop) if isinstance(n,ast.Subscript)
                and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                and n.value.func.attr=='groupby')
    rng=np.random.default_rng(917)
    keys=rng.integers(0,4,(800,len(roles)))
    values=rng.normal(20,3,800);values[::13]=np.nan
    times=rng.integers(0,90,800)
    frame=pd.DataFrame(keys,columns=roles)
    frame[target]=np.where(times<=60,values,np.nan)
    namespace={'grid_df':frame,'icols':icols,'np':np,'__builtins__':{}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),
                 '<pinned-numerical-loop>','exec'),namespace)
    actual=encode(values,times,keys,60,[[roles.index(v) for v in g] for g in icols])
    expected=frame.iloc[:,len(roles)+1:].to_numpy()
    np.testing.assert_equal(actual,expected)
    paths_to_hash=[ROOT/'sciona/m5_encoding.py',ROOT/'tests/test_m5_encoding.py',Path(__file__).resolve(),pins_path]
    return dict(status='passed',approved=False,catalog_mutations=0,
        checks=dict(synthetic_only=True,source_columns_compared=expected.shape[1],exact_float16=True),
        source_commit=pins['commit'],source_software_sha256=expected_hashes[0],
        scope='Pinned grouped moment loop only; fixed cutoff supplied independently. Full preprocessing remains incomplete.',
        sha256={str(p.relative_to(ROOT)):sha(p) for p in paths_to_hash})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    report=compare(args.source_root)
    (ROOT/'docs/reviews/competition_m5_encoding_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))

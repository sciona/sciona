"""Evaluate only the pinned downcast function on synthetic columns."""
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
from sciona.m5_precision import downcast


def compare(source_root):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pins_path=ROOT/'docs/reviews/competition_m5_source_pins.json'
    pins=json.loads(pins_path.read_text())
    sources=list(source_root.rglob('1. preprocessing.py'))
    hashes=[v for k,v in pins['files'].items() if k.endswith('1. preprocessing.py')]
    if len(sources)!=1 or len(hashes)!=1 or sha(sources[0])!=hashes[0]:
        raise ValueError('Unique pinned software required')
    tree=ast.parse(sources[0].read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reduce_mem_usage']
    assert len(selected)==1
    namespace={'np':np,'__builtins__':{'str':str,'__import__':__import__}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),
                 '<pinned-numerical-function>','exec'),namespace)
    cases=[]
    for dtype in (np.int16,np.int32,np.int64):
        for size in (np.int8,np.int16,np.int32):
            limits=np.iinfo(size)
            if limits.min>=np.iinfo(dtype).min and limits.max<=np.iinfo(dtype).max:
                cases.extend([np.array([limits.min],dtype=dtype),np.array([limits.max],dtype=dtype),
                              np.array([limits.min+1,limits.max-1],dtype=dtype)])
    for dtype in (np.float16,np.float32,np.float64):
        cases.extend([np.array([np.nan],dtype=dtype),np.array([np.inf],dtype=dtype),
                      np.array([-np.inf],dtype=dtype),np.array([np.nan,1.0001],dtype=dtype),
                      np.array([np.finfo(dtype).max],dtype=dtype)])
    rng=np.random.default_rng(512)
    cases.extend(rng.normal(0,10,50) for _ in range(100))
    for values in cases:
        expected=namespace['reduce_mem_usage'](pd.DataFrame({'value':values}),verbose=False)['value'].to_numpy()
        actual=downcast(values)
        assert actual.dtype==expected.dtype
        np.testing.assert_equal(actual,expected)
    paths=[ROOT/'sciona/m5_precision.py',ROOT/'tests/test_m5_precision.py',Path(__file__).resolve(),pins_path]
    return dict(status='passed',approved=False,catalog_mutations=0,source_commit=pins['commit'],
        source_software_sha256=hashes[0],checks=dict(synthetic_only=True,source_comparisons=len(cases),exact_values_and_dtypes=True),
        sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        scope='Column precision only; no full preprocessing or model qualification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    report=compare(parser.parse_args().source_root)
    (ROOT/'docs/reviews/competition_m5_precision_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))

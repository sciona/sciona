"""Compare MCC threshold selection with pinned source on synthetic cases."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.vsb_threshold import threshold
SOURCE_SHA='ee289da71520cfa781723070f5face629c962e9430fd06bb4f2069a3b72a83b9'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Source pin differs')
    functions=[]
    for c in json.loads(raw):
        if c['index']==73:
            for node in ast.parse(c['source']).body:
                if isinstance(node,ast.FunctionDef) and node.name in ('mcc','eval_mcc'):
                    node.decorator_list=[];functions.append(node)
    assert len(functions)==2
    ns={'np':np,'__builtins__':{'range':range}}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned numerical source>','exec'),ns)
    rng=np.random.default_rng(99);cases=[]
    for n in (1,2,40,200):
        for _ in range(100):cases.append((rng.integers(0,2,n),rng.integers(0,8,n)/7))
    cases.extend([(np.ones(20,int),np.linspace(0,1,20)),(np.zeros(20,int),np.linspace(0,1,20))])
    for y,p in cases:
        old=ns['eval_mcc'](y,p);new=threshold(y,p)
        assert old[0]==new['threshold']
        np.testing.assert_allclose(old[1],new['mcc'],atol=1e-15,rtol=1e-14)
    files=[ROOT/'sciona/vsb_threshold.py',ROOT/'tests/test_vsb_threshold.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,source_code_cells_sha256=SOURCE_SHA,
        comparison_cases=len(cases),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Numerical threshold comparison only. Source optimizes >= while downstream decisions use strict >; no unbiased validation claim.'])
    (ROOT/'docs/reviews/competition_vsb_threshold_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',comparison_cases=len(cases))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True)
    main(p.parse_args().source_code)

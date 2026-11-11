"""Pinned-source checkpoint metric comparison on synthetic tied predictions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.amex_metric import score
SOURCE_SHA='38291db3eca21f9e6eb8f87de166accbf8d178e68e2cdeaa8f0be3c158caaf70'


def main(path):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Source pin differs')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='amex_metric_mod'];assert len(nodes)==1
    ns={'np':np,'__builtins__':{'int':int}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned metric source>','exec'),ns)
    rng=np.random.default_rng(116)
    for _ in range(400):
        y=np.r_[0,1,rng.integers(0,2,98)];p=rng.integers(0,10,100)/9
        np.testing.assert_allclose(score(y,p),ns['amex_metric_mod'](y,p),atol=1e-14,rtol=1e-13)
    paths=[ROOT/'sciona/amex_metric.py',ROOT/'tests/test_amex_metric.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,comparison_cases=400,source_software_sha256=SOURCE_SHA,
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},limits=['Synthetic checkpoint metric comparison only; both classes explicitly required.'])
    (ROOT/'docs/reviews/competition_amex_metric_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='passed',comparison_cases=400)))


if __name__=='__main__':
    if not __debug__:raise RuntimeError('Assertions required')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True);main(p.parse_args().source_code)

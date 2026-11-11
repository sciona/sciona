"""Compare independent uncertainty objectives with pinned source on synthetic arrays."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.m5u_loss import pinball,weighted_scaled


def main(source):
    pins_path=ROOT/'docs/reviews/competition_m5_uncertainty_source_pins.json'
    pins=json.loads(pins_path.read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==next(v for k,v in pins['files'].items() if k.endswith('quantiles_sage_1to9_eval.py'))
    t=ast.parse(source.read_text());functions=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('quantile_loss','wspl')]
    assert len(functions)==2
    namespace={'np':np,'__builtins__':{}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'<pinned-numerical-functions>','exec'),namespace)
    rng=np.random.default_rng(617)
    for case in range(200):
        actual=rng.normal(size=31);predicted=rng.normal(size=31)
        weights=rng.uniform(.1,5,31);scale=rng.uniform(.1,3,31);q=rng.uniform(.001,.999)
        np.testing.assert_allclose(pinball(actual,predicted,q),namespace['quantile_loss'](actual,predicted,q),rtol=1e-14)
        np.testing.assert_allclose(weighted_scaled(actual,predicted,weights,scale,q),namespace['wspl'](actual,predicted,weights,scale,q),rtol=1e-14)
    paths=[ROOT/'sciona/m5u_loss.py',ROOT/'tests/test_m5u_loss.py',Path(__file__).resolve(),pins_path]
    report=dict(status='passed',approved=False,catalog_mutations=0,checks=dict(synthetic_only=True,source_comparisons=400),source_software_sha256=digest,
                sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},scope='Loss arithmetic only; full quantile training and hierarchy pipeline remain incomplete.')
    (ROOT/'docs/reviews/competition_m5_uncertainty_loss_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    main(parser.parse_args().source)

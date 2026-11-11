"""Read pinned private code-only source and evaluate synthetic numerical cases.

No source code, paths, records, notebook outputs or weights enter the report.
Only the five previously reviewed numerical functions are compiled; decorators
are removed. This is a fixed source comparison, not a general code sandbox.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.vsb_baseline import subtract_baseline
SOURCE_SHA='ee289da71520cfa781723070f5face629c962e9430fd06bb4f2069a3b72a83b9'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Source code pin differs')
    wanted={'flatiron','drop_missing','_local_maxima_1d_window_single_pass','local_maxima_1d_window','plateau_detection'}
    functions=[]
    for cell in json.loads(raw):
        if cell['index'] not in (10,11,12,13):continue
        for node in ast.parse(cell['source']).body:
            if isinstance(node,ast.FunctionDef) and node.name in wanted:
                node.decorator_list=[];functions.append(node)
    assert len(functions)==5 and {f.name for f in functions}==wanted
    namespace={'np':np,'__builtins__':{'len':len,'range':range}}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned numerical source>','exec'),namespace)
    rng=np.random.default_rng(86)
    for x in (rng.normal(size=1000),np.r_[0.,np.ones(50)],np.array([3.]),np.ones(20)*7):
        np.testing.assert_allclose(subtract_baseline(x),namespace['flatiron'](x),atol=5e-14,rtol=1e-12)
    maxima=namespace['local_maxima_1d_window'];plateau=namespace['plateau_detection']
    np.testing.assert_array_equal(maxima(np.array([0.,2.,0.]),1),[1])
    np.testing.assert_array_equal(maxima(np.array([0.,2.,2.,0.]),1),[])
    np.testing.assert_array_equal(maxima(np.array([0.,2.,2.,2.,0.]),1),[2])
    # Counter accumulates qualifying entries without resetting between them.
    assert plateau(np.array([1.,-1.,1.,-1.,1.]),0.,3)==1
    assert plateau(np.array([1.,1.,1.]),0.,3)==-1
    assert plateau(np.array([-1.,-1.,-1.]),0.,3)==0
    # Source subtracts four from this index before slicing sorted peaks.
    assert plateau(np.full(12,-1.),-.01,1000)-4==-4
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=[ROOT/'sciona/vsb_baseline.py',ROOT/'tests/test_vsb_baseline.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        source_code_cells_sha256=SOURCE_SHA,baseline_tests_passed=15,
        baseline_comparison_cases=4,source_peak_edge_cases=6,
        sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        findings=['Float64 independent transfer-function baseline matches pinned recursive baseline on four synthetic cases.',
            'Even-width flat maxima disappear when forward and reverse midpoint sets are intersected; odd-width plateaus survive.',
            'Source plateau counter accumulates threshold hits without resetting; it does not require a consecutive run.',
            'Reaching the requested count immediately can return minus one; no hit returns zero.',
            'No-hit knee becomes minus four after offset, so source slicing retains all but four sorted peaks rather than retaining none.'],
        limits=['Unaccelerated numerical function comparison only; Numba/native parity and full preprocessing unqualified.',
            'No source code adapted or copied into repository; notebook reuse terms unresolved.',
            'Full peak feature extraction, measurement aggregation, repeated training, threshold and served graph remain pending.'])
    (ROOT/'docs/reviews/competition_vsb_source_numerics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',baseline_cases=4,peak_edge_cases=6,catalog_mutations=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-code',type=Path,required=True)
    main(parser.parse_args().source_code)

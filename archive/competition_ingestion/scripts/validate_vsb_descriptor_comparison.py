"""Synthetic descriptor comparison at the pinned source's fixed signal length."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.vsb_descriptors import peak_descriptors
SOURCE_SHA='ee289da71520cfa781723070f5face629c962e9430fd06bb4f2069a3b72a83b9'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Pinned source differs')
    wanted={'clip','create_sawtooth_template','calculate_peak_features'};functions=[]
    for cell in json.loads(raw):
        if cell['index'] not in (15,16):continue
        for f in ast.parse(cell['source']).body:
            if isinstance(f,ast.FunctionDef) and f.name in wanted:
                f.decorator_list=[];functions.append(f)
    assert len(functions)==3 and {f.name for f in functions}==wanted
    ns={'np':np,'num_peak_features':4,'__builtins__':{'range':range}}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned numerical source>','exec'),ns)
    compare=ns['calculate_peak_features'];cases=0
    size=800000;rng=np.random.default_rng(92)
    for position in (0,1,4,25,size//2,size-26,size-5,size-1):
        for sign in (-1.,1.):
            x=rng.uniform(-.2,.2,size);x[position]=3*sign
            end=min(size,position+4)
            x[position:end]=sign*np.linspace(3,-3,4)[:end-position]
            positions=np.asarray([position],dtype=np.int64)
            expected=compare(positions,x);actual=peak_descriptors(positions,x)
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-13,equal_nan=True)
            cases+=1
    x=np.zeros(size);x[200]=1;x[201]=2
    try:compare(np.array([200]),x)
    except AssertionError:pass
    else:raise AssertionError('Source accepted unaligned maximum')
    try:peak_descriptors([200],x)
    except ValueError:pass
    else:raise AssertionError('Independent implementation accepted unaligned maximum')
    files=[ROOT/'sciona/vsb_descriptors.py',ROOT/'tests/test_vsb_descriptors.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        source_version_id='d79e40e4-45b0-5e74-a83c-c68c91f12afb',source_content_hash='565e511ee8d20ce0b32adf5c8cc726150f235b1f97d5502d62c96cfe339f5ac1',source_code_cells_sha256=SOURCE_SHA,
        checks=dict(descriptor_cases=cases,source_length_exercised=True,unaligned_maximum_rejected_by_both=True,missing_neighbor_nan_preserved=True),
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Fixed-length synthetic numerical source comparison only; no native Numba or historical precision qualification.',
            'Independent descriptor windows clip to actual input length; nominal small radius subtraction retained at boundaries.',
            'Neighbor NaN values require explicit downstream handling. No whole measurement, trainer or publication qualification.'])
    (ROOT/'docs/reviews/competition_vsb_descriptor_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True)
    main(p.parse_args().source_code)

"""Pinned-source comparisons for synthetic measurement features and Fourier phase."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.vsb_measurement import fundamental,phase_quadrants,aggregate
SOURCE_SHA='ee289da71520cfa781723070f5face629c962e9430fd06bb4f2069a3b72a83b9'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Pinned source differs')
    wanted={'process','calculate_50hz_fourier_coefficient'};functions=[]
    for cell in json.loads(raw):
        if cell['index'] not in (20,50):continue
        for f in ast.parse(cell['source']).body:
            if isinstance(f,ast.FunctionDef) and f.name in wanted:
                f.decorator_list=[];functions.append(f)
    assert len(functions)==2 and {f.name for f in functions}==wanted
    ns={'np':np,'pd':pd,'numba':SimpleNamespace(prange=range),'USE_SIMPLIFIED_VERSION':False,'__builtins__':{}}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned numerical source>','exec'),ns)
    rng=np.random.default_rng(95)
    for count in (0,1,250):
        groups=rng.integers(0,5,count);heights=rng.uniform(1,80,count);f=rng.uniform(0,1,(count,4));q=rng.integers(-1,4,count)
        if count:f[0,:2]=np.nan
        peaks=pd.DataFrame(dict(id_measurement=groups,px=np.arange(count),height=heights,ratio_next=f[:,0],ratio_prev=f[:,1],small_dist_to_min=f[:,2],sawtooth_rmse=f[:,3],Q=q))
        meta=pd.DataFrame(dict(id_measurement=np.arange(6)))
        expected=ns['process'](peaks,meta).to_numpy()
        actual=aggregate(groups,heights,f,q,measurement_count=6)
        np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-13,equal_nan=True)
    x=rng.normal(size=(800000,3))
    expected=ns['calculate_50hz_fourier_coefficient'](x);actual=fundamental(x)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=1e-12)
    positions=np.array([0,1,200000,400000,600000,799999])
    for coefficient in (1+0j,-1j,1j,-1+0j):
        degrees=(np.degrees(2*np.pi*50*positions*(.02/800000)+np.angle(coefficient))+90)%360
        source=pd.cut(degrees,[0,90,180,270,360],labels=[0,1,2,3]).codes
        np.testing.assert_array_equal(phase_quadrants(positions,coefficient,length=800000),source)
    files=[ROOT/'sciona/vsb_measurement.py',ROOT/'tests/test_vsb_measurement.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        source_version_id='d79e40e4-45b0-5e74-a83c-c68c91f12afb',source_content_hash='565e511ee8d20ce0b32adf5c8cc726150f235b1f97d5502d62c96cfe339f5ac1',source_code_cells_sha256=SOURCE_SHA,
        checks=dict(aggregate_cases=3,aggregate_width=9,source_length_fourier_channels=3,phase_boundary_cases=4),
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Synthetic comparison to unaccelerated source only; FFT changes summation order and near-boundary floating-point classification may differ.',
            'Input-length generalization interprets the fundamental as one cycle across each supplied signal.',
            'NaN missing aggregates intentionally retained for native learner handling; full training and publication remain pending.'])
    (ROOT/'docs/reviews/competition_vsb_measurement_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True)
    main(p.parse_args().source_code)

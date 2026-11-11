"""Compare independent peak selection to pinned private code on synthetic inputs."""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.vsb_peaks import window_maxima,plateau_location,select_peaks
SOURCE_SHA='ee289da71520cfa781723070f5face629c962e9430fd06bb4f2069a3b72a83b9'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Pinned source differs')
    wanted={'flatiron','drop_missing','_local_maxima_1d_window_single_pass','local_maxima_1d_window','plateau_detection','get_peaks'}
    functions=[]
    for cell in json.loads(raw):
        if cell['index'] not in (10,11,12,13,14):continue
        for f in ast.parse(cell['source']).body:
            if isinstance(f,ast.FunctionDef) and f.name in wanted:
                f.decorator_list=[];functions.append(f)
    assert len(functions)==6 and {f.name for f in functions}==wanted
    ns={'np':np,'__builtins__':{'len':len,'range':range}}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned numerical source>','exec'),ns)
    peak_cases=0
    for size in range(1,8):
        for seq in itertools.product((0.,1.,2.),repeat=size):
            x=np.asarray(seq)
            for window in (1,2,5):
                np.testing.assert_array_equal(window_maxima(x,window=window),ns['local_maxima_1d_window'](x,window))
                peak_cases+=1
    rng=np.random.default_rng(88)
    for _ in range(100):
        g=rng.normal(size=100)
        for count in (1,5,120):
            assert plateau_location(g,threshold=0,count=count)==ns['plateau_detection'](g,0,count)
    full_cases=0
    for size in (512,4096,12000):
        for window in (1,25):
            x=rng.normal(size=size)
            old=ns['get_peaks'](x,window=window);new=select_peaks(x,window=window)
            np.testing.assert_array_equal(new['positions'],old[0])
            np.testing.assert_allclose(new['heights'],old[1],atol=5e-14,rtol=1e-12)
            np.testing.assert_allclose(new['residual'],old[2],atol=5e-14,rtol=1e-12)
            full_cases+=1
    files=[ROOT/'sciona/vsb_baseline.py',ROOT/'sciona/vsb_peaks.py',ROOT/'tests/test_vsb_baseline.py',ROOT/'tests/test_vsb_peaks.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        source_version_id='d79e40e4-45b0-5e74-a83c-c68c91f12afb',source_content_hash='565e511ee8d20ce0b32adf5c8cc726150f235b1f97d5502d62c96cfe339f5ac1',source_code_cells_sha256=SOURCE_SHA,
        checks=dict(exhaustive_directional_cases=peak_cases,threshold_counter_cases=300,full_selection_cases=full_cases),
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Independent floating-point implementation compared to unaccelerated source functions on synthetic signals; native Numba and historical numerical parity unqualified.',
            'Fewer than two candidate peaks explicitly reject because source convolution has an empty gradient.',
            'Source tie sorting and negative knee slicing retained. No full feature, trainer or publication qualification.'])
    (ROOT/'docs/reviews/competition_vsb_peak_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True)
    main(p.parse_args().source_code)

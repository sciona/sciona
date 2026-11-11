"""Source polygon augmentation parity using synthetic68-point predictor outputs."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import skimage.draw
from skimage import measure

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import dfdc_convex_hull as implementation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    path=args.source_root/'training/datasets/classifier_dataset.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==next(r['sha256'] for r in pins['files'] if r['path']=='training/datasets/classifier_dataset.py')
    original=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='blackout_convex_hull')
    current=next(n for n in ast.parse(Path(implementation.__file__).read_text()).body if isinstance(n,ast.FunctionDef))
    original_body=ast.dump(ast.Module(body=original.body,type_ignores=[]))
    assert original_body==ast.dump(ast.Module(body=current.body,type_ignores=[]))
    assert [a.arg for a in current.args.args]==['img','detector','predictor']
    namespace={'np':np,'random':random,'skimage':skimage,'measure':measure}
    exec(compile(ast.Module(body=[original],type_ignores=[]),'<source-convex-hull>','exec'),namespace)
    order=list(range(17))+list(range(26,16,-1))
    theta=np.linspace(0,2*np.pi,27,endpoint=False)
    points=np.zeros((68,2),dtype=int)
    points[order]=np.stack([14+10*np.cos(theta),13+9*np.sin(theta)],axis=1).astype(int)
    detector=lambda image:[object()]
    predictor=lambda image,rectangle:SimpleNamespace(parts=lambda:[SimpleNamespace(x=int(x),y=int(y)) for x,y in points])
    namespace.update(detector=detector,predictor=predictor)
    cases=0;initial=random.getstate()
    try:
        for shape in ((29,31,3),(31,29,3)):
            image=np.random.default_rng(729).integers(1,256,shape,dtype=np.uint8)
            for seed in range(16):
                a,b=image.copy(),image.copy()
                random.seed(seed);expected=namespace['blackout_convex_hull'](a);state=random.getstate()
                random.seed(seed);actual=implementation.blackout_convex_hull(b,detector,predictor)
                assert actual is expected is None
                np.testing.assert_array_equal(a,b)
                assert random.getstate()==state
                assert np.any(b==0) and np.any(b!=0)
                cases+=1
        failures=0
        for detector,predictor in ((lambda image:[],predictor),
                                   (lambda image:[object()],lambda *args:(_ for _ in ()).throw(RuntimeError('synthetic error')))):
            namespace.update(detector=detector,predictor=predictor)
            a,b=image.copy(),image.copy()
            random.seed(90);namespace['blackout_convex_hull'](a);state=random.getstate()
            random.seed(90);implementation.blackout_convex_hull(b,detector,predictor)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,image)
            assert random.getstate()==state
            failures+=1
    finally:random.setstate(initial)
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_dfdc_convex_hull.py'],cwd=ROOT,check=True)
    files=['sciona/dfdc_convex_hull.py','tests/test_dfdc_convex_hull.py',
           'scripts/validate_dfdc_convex_hull.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-convex-hull-validation.v1','result':'passed','source_commit':pins['commit'],
        'checks':{'source_body_ast_equal':True,'source_image_rng_cases':cases,
                  'source_failure_cases':failures,'independent_geometry_failure_tests':7},
        'dependency_state':{'dlib_installed':importlib.util.find_spec('dlib') is not None,'predictor_weights_loaded':False},
        'semantics':['Ordered source jaw/brow polygon rasterized, centroid truncated to int, one half zeroed in-place.',
                     'Two strict >0.5 draws select half/direction; function returns None.',
                     'Source caught detector/predictor/polygon failures leave the image unchanged in tested cases.'],
        'adaptation':'Detector and predictor passed explicitly instead of global pretrained objects; function body unchanged.',
        'limits':'Synthetic68-point predictor outputs only. Actual dlib detector/predictor execution and explicit model-state handling remain outstanding. No pretrained landmarks, quality or full preparation/promotion claim.',
        'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_convex_hull.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

"""Synthetic geometry and RNG parity for pinned five-point landmark removal."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import dfdc_landmark_removal as implementation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    path=args.source_root/'training/datasets/classifier_dataset.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==next(r['sha256'] for r in pins['files'] if r['path']=='training/datasets/classifier_dataset.py')
    names={'dist','remove_eyes','remove_nose','remove_mouth','remove_landmark'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    current={n.name:n for n in ast.parse(Path(implementation.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    assert len(nodes)==5
    for n in nodes:assert ast.dump(n)==ast.dump(current[n.name])
    namespace={'math':math,'random':random,'cv2':cv2,'np':np,'binary_dilation':binary_dilation}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-landmark-removal>','exec'),namespace)
    points=[np.array(x,dtype=np.int32) for x in (
        [[5,5],[13,5],[9,10],[6,14],[12,14]],
        [[5,5],[6,5],[5,7],[5,9],[6,9]],
        [[-2,3],[25,4],[11,9],[3,18],[17,17]],
    )]
    direct=0;stochastic=0;initial=random.getstate()
    try:
        for shape in ((21,21,3),(21,29,3),(29,21,3)):
            image=np.random.default_rng(819).integers(1,256,shape,dtype=np.uint8)
            for landmarks in points:
                for name in ('remove_eyes','remove_nose','remove_mouth'):
                    a=image.copy();b=image.copy()
                    expected=namespace[name](a,landmarks.copy())
                    actual=getattr(implementation,name)(b,landmarks.copy())
                    np.testing.assert_array_equal(actual,expected)
                    np.testing.assert_array_equal(a,image);np.testing.assert_array_equal(b,image)
                    direct+=1
            for seed in range(16):
                random.seed(seed);expected=namespace['remove_landmark'](image.copy(),points[0]);state=random.getstate()
                random.seed(seed);actual=implementation.remove_landmark(image.copy(),points[0])
                np.testing.assert_array_equal(actual,expected)
                assert random.getstate()==state
                stochastic+=1
    finally:random.setstate(initial)
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_dfdc_landmark_removal.py'],cwd=ROOT,check=True)
    files=['sciona/dfdc_landmark_removal.py','tests/test_dfdc_landmark_removal.py',
           'scripts/validate_dfdc_landmark_removal.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-landmark-removal-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'source_ast_equalities':5,'direct_source_geometry_cases':direct,
                 'stochastic_source_output_rng_cases':stochastic,'independent_tests':8},
       'semantics':['Eyes/nose/mouth removals copy input; no-op random branch returns original input.',
                    'Sequential strict >0.5 draws give probabilities 1/2 eyes,1/4 mouth,1/8 nose,1/8 unchanged.',
                    'Dilation floor-divides landmark separation; zero iterations retains SciPy convergence behavior.',
                    'These are five-point preprocessing landmarks; convex-hull augmentation still requires separate68-point prediction.'],
       'limits':'Synthetic landmarks and image arrays only. No landmark detector weights, prediction quality, convex-hull branch or full preparation/lifecycle approval claimed.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_landmark_removal.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

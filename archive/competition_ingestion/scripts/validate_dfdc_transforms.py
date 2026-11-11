"""Source transform definitions, actual synthetic image-mask execution and geometry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import sys

import albumentations as A
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import dfdc_transforms as implementation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    nodes=[]
    for name,names in [('training/transforms/albu.py',{'isotropically_resize_image','IsotropicResize'}),
                       ('training/pipelines/train_classifier.py',{'create_train_transforms','create_val_transforms'})]:
        path=args.source_root/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']==name)
        nodes.extend(n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
    current={n.name:n for n in ast.parse(Path(implementation.__file__).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for n in nodes:assert ast.dump(n)==ast.dump(current[n.name])
    namespace=dict(vars(A));namespace['cv2']=cv2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-dfdc-transforms>','exec'),namespace)
    rng=np.random.default_rng(625)
    cases=0;initial_py=random.getstate();initial_np=np.random.get_state()
    try:
        for name in ('create_train_transforms','create_val_transforms'):
            for h,w in ((47,73),(73,47),(63,63)):
                image=rng.integers(0,256,(h,w,3),dtype=np.uint8)
                mask=(rng.integers(0,2,(h,w),dtype=np.uint8)*255)
                source=namespace[name](380);adapted=getattr(implementation,name)(380)
                for seed in range(12):
                    random.seed(seed);np.random.seed(seed)
                    expected=source(image=image.copy(),mask=mask.copy())
                    py_state=random.getstate();np_state=np.random.get_state()
                    random.seed(seed);np.random.seed(seed)
                    actual=adapted(image=image.copy(),mask=mask.copy())
                    for key in ('image','mask'):np.testing.assert_array_equal(actual[key],expected[key])
                    assert random.getstate()==py_state
                    state=np.random.get_state();assert state[0]==np_state[0] and state[2:]==np_state[2:]
                    np.testing.assert_array_equal(state[1],np_state[1])
                    assert actual['image'].shape==(380,380,3) and actual['mask'].shape==(380,380)
                    assert set(np.unique(actual['mask'])).issubset({0,255})
                    cases+=1
    finally:random.setstate(initial_py);np.random.set_state(initial_np)
    # Independent nearest-neighbor image/mask doubling oracle for the custom transform.
    image=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
    mask=(np.indices((4,6)).sum(0)%2).astype(np.uint8)
    resize=implementation.IsotropicResize(12,interpolation_up=cv2.INTER_NEAREST)
    actual=resize(image=image,mask=mask)
    nearest=np.repeat(np.repeat(image,2,0),2,1)
    np.testing.assert_array_equal(resize.apply(image,interpolation_up=cv2.INTER_NEAREST),nearest)
    np.testing.assert_array_equal(actual['image'],cv2.resize(image,(12,8),interpolation=cv2.INTER_CUBIC))
    assert not np.array_equal(actual['image'],nearest)
    np.testing.assert_array_equal(actual['mask'],np.repeat(np.repeat(mask,2,0),2,1))
    # Source padding puts odd spare pixel on bottom/right; no intensity change for constants.
    image=np.full((3,6,3),89,dtype=np.uint8);mask=np.ones((3,6),dtype=np.uint8)
    got=implementation.create_val_transforms(9)(image=image,mask=mask)
    expected=np.zeros((9,9,3),dtype=np.uint8);expected[2:6]=89
    expected_mask=np.zeros((9,9),dtype=np.uint8);expected_mask[2:6]=1
    np.testing.assert_array_equal(got['image'],expected);np.testing.assert_array_equal(got['mask'],expected_mask)
    files=['sciona/dfdc_transforms.py','scripts/validate_dfdc_transforms.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-transforms-validation.v1','result':'passed','source_commit':pins['commit'],
       'dependency':{'albumentations':A.__version__,'opencv':cv2.__version__},
       'checks':{'source_ast_equalities':4,'actual_source_image_mask_rng_cases':cases,
                 'independent_direct_nearest_resize':True,'constructor_interpolation_ignored':True,'independent_padding':True},
       'semantics':['Source train and validation order/probabilities retained; B7 size380 passed explicitly.',
                    'Spatial augmentation applies jointly to image and mask; mask resize uses nearest interpolation.',
                    'Constructor interpolation_up/down attributes are not forwarded to image apply under installed library: default area/cubic is effective; explicit direct apply arguments work.',
                    'Factory source default300 retained but does not override B7 configuration.'],
       'limits':'Actual synthetic transforms under installedAlbumentations1.3.0, compared with same-library source definitions. SourceDocker pins1.0.0; historical stochastic/binary equivalence is not asserted. Landmark/region augmentation and tensor preparation still need connection; no full lifecycle or promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_transforms.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

"""Source __getitem__ parity with synthetic in-memory file boundaries."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import cv2
import numpy as np
import torch
from albumentations.pytorch.functional import img_to_tensor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_preparation import prepare_sample
from sciona.dfdc_landmark_removal import remove_landmark
from sciona.dfdc_convex_hull import blackout_convex_hull
from sciona.dfdc_occlusion_masks import prepare_bit_masks
from sciona.dfdc_transforms import create_train_transforms,create_val_transforms


class NumpyProxy:
    def __init__(self,landmarks):self.landmarks=landmarks
    def load(self,_):return self.landmarks.copy()
    def __getattr__(self,key):return getattr(np,key)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    p=args.source_root/'training/datasets/classifier_dataset.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']=='training/datasets/classifier_dataset.py')
    cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    points=np.zeros((68,2),dtype=int);order=list(range(17))+list(range(26,16,-1))
    theta=np.linspace(0,2*np.pi,27,endpoint=False)
    points[order]=np.stack([20+12*np.cos(theta),20+12*np.sin(theta)],axis=1).astype(int)
    detector=lambda image:[object()]
    predictor=lambda image,rectangle:SimpleNamespace(parts=lambda:[SimpleNamespace(x=int(x),y=int(y)) for x,y in points])
    def unexpected_failure(*args,**kwargs):raise AssertionError('original preparation raised unexpectedly')
    image=np.random.default_rng(635).integers(1,256,(47,61,3),dtype=np.uint8)
    landmarks=np.array([[12,12],[32,12],[22,23],[15,32],[29,32]],dtype=np.int32)
    namespace={'random':random,'img_to_tensor':img_to_tensor,'remove_landmark':remove_landmark,
      'blackout_convex_hull':lambda image:blackout_convex_hull(image,detector,predictor),
      'prepare_bit_masks':prepare_bit_masks,'sys':sys,'traceback':SimpleNamespace(print_exc=unexpected_failure)}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<source-preparation>','exec'),namespace)
    cases=0;initial_py=random.getstate();initial_np=np.random.get_state()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            for mode in ('train','val'):
                factory=create_train_transforms if mode=='train' else create_val_transforms
                for label in (0,1):
                    for mask in (None,np.full(image.shape[:2],255,dtype=np.uint8)):
                        for available in (False,True):
                            namespace['np']=NumpyProxy(landmarks)
                            namespace['os']=SimpleNamespace(path=SimpleNamespace(join=os.path.join,exists=lambda _:available))
                            namespace['cv2']=SimpleNamespace(IMREAD_COLOR=cv2.IMREAD_COLOR,IMREAD_GRAYSCALE=cv2.IMREAD_GRAYSCALE,
                                COLOR_BGR2RGB=cv2.COLOR_BGR2RGB,cvtColor=cv2.cvtColor,
                                imread=lambda path,flag:image[:,:,::-1].copy() if flag==cv2.IMREAD_COLOR else (None if mask is None else mask.copy()))
                            original=SimpleNamespace(data=[('','',label,'',0,0)],mode=mode,label_smoothing=.01,
                                data_root='',crops_dir='',hardcore=True,rotation=False,padding_part=3,
                                transforms=factory(380),normalize={'mean':[.485,.456,.406],'std':[.229,.224,.225]})
                            adapted_transform=factory(380)
                            for seed in [*range(12),15,42,69]:
                                random.seed(seed);np.random.seed(seed)
                                with redirect_stdout(io.StringIO()):expected=namespace['__getitem__'](original,0)
                                state=random.getstate();nstate=np.random.get_state()
                                random.seed(seed);np.random.seed(seed)
                                actual=prepare_sample(image,label,mode=mode,mask=mask,landmarks=landmarks if available else None,
                                    transforms=adapted_transform,detector=detector,predictor=predictor)
                                torch.testing.assert_close(actual['image'],expected['image'],rtol=0,atol=0)
                                np.testing.assert_array_equal(actual['labels'],expected['labels'])
                                assert actual['valid']==expected['valid'] and actual['rotations']==0
                                assert random.getstate()==state
                                current=np.random.get_state();assert current[0]==nstate[0] and current[2:]==nstate[2:]
                                np.testing.assert_array_equal(current[1],nstate[1])
                                assert 'img_name' not in actual and actual['image'].shape==(3,380,380)
                                cases+=1
    finally:random.setstate(initial_py);np.random.set_state(initial_np)
    # Independent normalization and validity boundaries with augmentation disabled by val mode.
    normal=np.zeros((9,11,3),dtype=np.uint8);normal[:]=[0,128,255]
    got=prepare_sample(normal,1,mode='val',transforms=None,detector=None,predictor=None)
    expected=torch.tensor([(0/255-.485)/.229,(128/255-.456)/.224,(1-.406)/.225],dtype=torch.float32)
    torch.testing.assert_close(got['image'][:,0,0],expected,rtol=1e-6,atol=1e-7)
    assert got['valid']==0 and got['labels'].tolist()==[1]
    for count in (32,33):
        mask=np.zeros((9,11),dtype=np.uint8);mask.flat[:count]=21
        got=prepare_sample(normal,1,mode='val',mask=mask,transforms=None,detector=None,predictor=None)
        assert got['valid']==int(count>32)
    retry_cases=0
    before=random.getstate()
    try:
        for label,attempts in ((0,1),(1,5)):
            random.seed(15)  # Select region mask with absent five-point landmarks.
            with patch.object(random,'choice',wraps=random.choice) as choice:
                prepare_sample(image,label,mode='train',transforms=None,detector=detector,predictor=predictor)
                assert choice.call_count==attempts
            retry_cases+=1
    finally:random.setstate(before)
    files=['sciona/dfdc_preparation.py','sciona/dfdc_transforms.py','sciona/dfdc_landmark_removal.py',
           'sciona/dfdc_convex_hull.py','sciona/dfdc_occlusion_masks.py','scripts/validate_dfdc_preparation.py',
           'docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-preparation-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'source_sample_output_rng_cases':cases,'independent_normalization':True,'independent_validity_boundaries':2,'independent_region_retry_cases':retry_cases},
       'semantics':['Source conditional random draws and five-attempt region-mask retry retained.',
                    'Missing difference mask becomes zeros; train labels clip to .01/.99.',
                    'Validity computed before final transforms with source strict pixel-count/value thresholds.',
                    'Active B7 rotation=False/padding_part3/hardcoreTrue configuration retained.'],
       'limits':'Source __getitem__ uses synthetic in-memory I/O and previously source-validated augmentation components. Actual dlib68-point predictor remains outstanding. No broken-file retries or media identities; malformed explicit inputs rejected. No original media, full lifecycle or promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

"""Original crop writer oracle with synthetic in-memory media boundaries."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_training_crops import build_preprocessing_detector,detect_training_boxes,crop_pair


class Capture:
    def __init__(self,rgb):self.rgb=rgb;self.index=-1
    def get(self,_):return len(self.rgb)
    def grab(self):self.index+=1;return self.index<len(self.rgb)
    def retrieve(self):return True,self.rgb[self.index,:,:,::-1].copy()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    for name in ('preprocessing/extract_crops.py','preprocessing/face_detector.py','preprocessing/generate_landmarks.py'):
        path=args.source_root/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']==name)
    fn=next(n for n in ast.parse((args.source_root/'preprocessing/extract_crops.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='extract_video')
    namespace={'json':json,'os':SimpleNamespace(path=os.path,makedirs=lambda *a,**k:None)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-crop-writer>','exec'),namespace)
    cases=0
    for h,w in ((80,96),(96,80)):
        rng=np.random.default_rng(527)
        original=rng.integers(0,256,(37,h,w,3),dtype=np.uint8)
        altered=rng.integers(0,256,(25,h,w,3),dtype=np.uint8)
        for initial in (None,[],[[4.2,5.3,17.5,19.6],[1.,1.,9.,8.],[20.,10.,34.,26.]]):
            boxes={0:initial,5:[[2.,2.,10.,10.]],10:None,20:[[3.,4.,13.,14.]],30:initial}
            actual=crop_pair(original,altered,boxes)
            for key,rgb in (('original',original),('altered',altered)):
                saved=[]
                def write(path,image):
                    assert image.size>0
                    frame,actor=map(int,Path(path).stem.split('_'))
                    saved.append({'frame':frame,'actor':actor,'image':image[:,:,::-1].copy()})
                    return True
                namespace['open']=lambda *args:io.StringIO(json.dumps(boxes))
                namespace['cv2']=SimpleNamespace(VideoCapture=lambda _:Capture(rgb),CAP_PROP_FRAME_COUNT=cv2.CAP_PROP_FRAME_COUNT,imwrite=write)
                namespace['extract_video'](('synthetic.mp4',''),'', '')
                assert len(saved)==len(actual[key])
                for a,b in zip(actual[key],saved):
                    assert (a['frame'],a['actor'])==(b['frame'],b['actor'])
                    np.testing.assert_array_equal(a['image'],b['image'])
                    assert a['frame']%10==0
                cases+=1
    # Independent slice oracle, including source reuse on altered pixel values.
    a=np.full((11,24,30,3),31,dtype=np.uint8);b=np.full_like(a,79)
    got=crop_pair(a,b,{0:[[4.,3.,10.,9.]]})
    assert got['original'][0]['image'].shape==(20,20,3)
    assert np.all(got['original'][0]['image']==31) and np.all(got['altered'][0]['image']==79)
    torch.set_num_threads(2)
    with patch('torch.load',side_effect=AssertionError('implicit state load')):
        detector=build_preprocessing_detector(stage='boxes',initialization='random',seed=628)
        assert detector.thresholds==[.85,.95,.95]
        with torch.no_grad():
            for p in detector.parameters():p.zero_()
            for model,head in ((detector.pnet,'conv4_1'),(detector.rnet,'dense5_1'),(detector.onet,'dense6_1')):
                getattr(model,head).bias.copy_(torch.tensor([-2.,2.]))
            detector.onet.dense6_3.bias.fill_(.5)
        seen=[];original_detect=detector.detect
        def detect(images,**kwargs):seen.append(len(images));return original_detect(images,**kwargs)
        detector.detect=detect
        frames=np.random.default_rng(629).integers(1,256,(35,80,80,3),dtype=np.uint8)
        boxes=detect_training_boxes(frames,detector)
        assert seen==[32,3] and list(boxes)==list(range(35)) and all(boxes.values())
        crops=crop_pair(frames,255-frames,boxes)
        assert {x['frame'] for x in crops['original']}=={0,10,20,30}
        assert len(crops['original'])==len(crops['altered'])>0
        landmark_detector=build_preprocessing_detector(stage='landmarks',initialization='random',seed=630)
        assert landmark_detector.thresholds==[.65,.75,.75]
    # Confirm distinct threshold literals from source constructor calls.
    for filename,expected in [('preprocessing/face_detector.py',[.85,.95,.95]),('preprocessing/generate_landmarks.py',[.65,.75,.75])]:
        calls=[n for n in ast.walk(ast.parse((args.source_root/filename).read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='MTCNN']
        assert len(calls)==1
        assert ast.literal_eval(next(k.value for k in calls[0].keywords if k.arg=='thresholds'))==expected
    files=['sciona/dfdc_training_crops.py','sciona/dfdc_detector.py','sciona/dfdc_faces.py',
           'sciona/dfdc_mtcnn_networks.py','sciona/dfdc_mtcnn_cascade.py','scripts/validate_dfdc_training_crops.py',
           'docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-training-crops-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'source_crop_writer_cases':cases,'independent_pair_slice':True,
                 'actual_training_detector_frames':35,'actual_detector_batches':[32,3],
                 'training_landmark_threshold_source_checks':2},
       'semantics':['Detect all original frames at half resolution in batches32; training thresholds differ from inference.',
                    'Every tenth frame cropped with original detections reused for altered clip; all actors retained.',
                    'No two-actor/320frame cap applied here: those limits belong to later landmark/difference generation.'],
       'limits':'Complete synthetic decoded arrays only. Partial decoding/indexing and media codec not covered. Actual detector uses constructed synthetic states, not pretrained-quality evidence. Landmark extraction and group folds still outstanding; no full lifecycle/promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_training_crops.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

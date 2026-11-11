"""Source prediction comparison and actual MTCNN-to-B7 synthetic integration."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Normalize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_classifier import build_classifier
from sciona.dfdc_detector import build_detector
from sciona.dfdc_frame_sampling import select_frames
from sciona.dfdc_prediction import predict_video


class FixedDetector:
    def __init__(self,count):self.count=count
    def detect(self,image,*,landmarks):
        if self.count==0:return None,None
        return np.array([[2.,2.,10.,10.]]*self.count),np.ones(self.count)


class FixedModel(torch.nn.Module):
    def __init__(self,probability):super().__init__();self.probability=probability
    def forward(self,x):return torch.logit(torch.full((len(x),1),self.probability,dtype=x.dtype))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    source=args.source_root/'kernel_utils.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==next(r['sha256'] for r in pins['files'] if r['path']=='kernel_utils.py')
    tree=ast.parse(source.read_text())
    names={'FaceExtractor','confident_strategy','put_to_center','isotropically_resize_image','predict_on_video'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    namespace={'np':np,'cv2':cv2,'Image':Image,'os':os,
      'torch':SimpleNamespace(tensor=lambda x,device:torch.tensor(x,device='cpu'),sigmoid=torch.sigmoid,no_grad=torch.no_grad),
      'normalize_transform':Normalize([.485,.456,.406],[.229,.224,.225])}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-dfdc-prediction>','exec'),namespace)
    def original_score(rgb,detector,models,frames_per_video):
        extractor=namespace['FaceExtractor'].__new__(namespace['FaceExtractor'])
        extractor.detector=detector
        extractor.video_read_fn=lambda _:select_frames(rgb,frames_per_video)
        with redirect_stdout(io.StringIO()):
            return namespace['predict_on_video'](extractor,'',frames_per_video,380,models,namespace['confident_strategy'])
    torch.set_num_threads(2)
    rgb=np.random.default_rng(918).integers(0,256,(1,80,80,3),dtype=np.uint8)
    source_cases=0
    for faces in (0,1,2,3,4,8):
        models=[FixedModel(.75).eval(),FixedModel(.125).eval()]
        expected=original_score(rgb,FixedDetector(faces),models,1)
        actual=predict_video(rgb,FixedDetector(faces),models,frames_per_video=1)
        assert actual['score']==float(expected)
        assert actual['classified_faces']==min(faces,3)
        assert actual['status']==('predicted' if faces>=2 else 'source_fallback')
        assert abs(actual['score']-(.4375 if faces>=2 else .5))<.001
        source_cases+=1
    # Seven distinct model probabilities exercise confidence-then-ensemble order.
    models=[FixedModel(p).eval() for p in (.15,.25,.35,.45,.65,.85,.95)]
    expected=original_score(rgb,FixedDetector(3),models,1)
    actual=predict_video(rgb,FixedDetector(3),models,frames_per_video=1)
    assert actual['score']==float(expected) and actual['models']==7
    source_cases+=1
    with warnings.catch_warnings(),patch('torch.load',side_effect=AssertionError('implicit weight load')):
        warnings.simplefilter('ignore',UserWarning)
        detector=build_detector(initialization='random',seed=617)
        with torch.no_grad():
            for p in detector.parameters():p.zero_()
            for model,head in [(detector.pnet,'conv4_1'),(detector.rnet,'dense5_1'),(detector.onet,'dense6_1')]:
                getattr(model,head).bias.copy_(torch.tensor([-2.,2.]))
            detector.onet.dense6_3.bias.fill_(.5)
        classifier=build_classifier(initialization='random',seed=923).eval().half()
        full_rgb=np.random.default_rng(924).integers(0,256,(32,40,40,3),dtype=np.uint8)
        batch_sizes=[]
        hook=classifier.register_forward_pre_hook(lambda module,args:batch_sizes.append(tuple(args[0].shape)))
        try:
            expected=original_score(full_rgb,detector,[classifier],32)
            actual=predict_video(full_rgb,detector,[classifier])
        finally:hook.remove()
        assert len(batch_sizes)==2 and batch_sizes[0]==batch_sizes[1]
        assert batch_sizes[0][0]>=2 and batch_sizes[0][1:]==(3,380,380)
        assert actual['status']=='predicted' and actual['score']==float(expected)
        assert actual['classified_faces']==batch_sizes[0][0]
    empty=predict_video(rgb[:0],FixedDetector(2),[FixedModel(.75).eval()])
    assert empty['reason']=='no_frames' and empty['score']==.5
    rejected=0
    for models,precision in (([], 'float16'),([FixedModel(.75)],'float16'),([FixedModel(.75).eval()],'other')):
        try:predict_video(rgb,FixedDetector(2),models,precision=precision)
        except ValueError:rejected+=1
        else:raise AssertionError('invalid configuration accepted')
    try:predict_video(rgb,FixedDetector(2),[FixedModel(float('nan')).eval()],frames_per_video=1)
    except ValueError:rejected+=1
    else:raise AssertionError('nonfinite score accepted')
    files=['sciona/dfdc_prediction.py','sciona/dfdc_frame_sampling.py','sciona/dfdc_faces.py',
           'sciona/dfdc_classifier.py','sciona/dfdc_stochastic_depth.py','docs/reviews/competition_dfdc_stochastic_depth.json','sciona/dfdc_detector.py','sciona/dfdc_mtcnn_networks.py',
           'sciona/dfdc_mtcnn_cascade.py','sciona/dfdc_inference_primitives.py',
           'scripts/validate_dfdc_prediction.py','docs/reviews/competition_dfdc_source_pins.json',
           'docs/reviews/competition_dfdc_facenet_source_pins.json']
    report={'format':'dfdc-prediction-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'source_standin_ensemble_cases':source_cases,'actual_detector_classifier_source_parity':True,
                 'actual_classifier_batch_shape':list(batch_sizes[0]),'empty_clip_fallback':True,'configuration_and_nonfinite_rejections':rejected},
       'adaptations':['Caller-decoded RGB, CPU tensor allocation, explicit model/detector initialization.',
                      'Fallback score .5 carries explicit status/reason instead of printing source error logs.',
                      'Empty models, training-mode models, invalid precision and nonfinite final score rejected.'],
       'limits':'Actual cascade-to-full-B7 inference used constructed synthetic detector states and random half-precision classifier, one checkpoint only. Seven-model orchestration uses stand-ins. Source comparison shares previously independently verified array frame selector. No media codec, pretrained quality, original checkpoint ensemble, training lifecycle or graph promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_prediction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

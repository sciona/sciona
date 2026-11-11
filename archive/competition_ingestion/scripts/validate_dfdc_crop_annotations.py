"""Sparse annotation source oracles and actual current-MTCNN rounding adaptation."""
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

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_crop_annotations import annotate_pair
from sciona.dfdc_training_crops import build_preprocessing_detector


class NumpyProxy:
    def __init__(self,saved):self.saved=saved
    def save(self,path,values):self.saved[tuple(map(int,Path(path).stem.split('_')))]=values.copy()
    def __getattr__(self,name):return getattr(np,name)


class Detector:
    def __init__(self,points):self.points=points;self.calls=0
    def detect(self,image,*,landmarks):
        assert landmarks;self.calls+=1
        return None,None,self.points


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    nodes=[]
    for path,name in [('preprocessing/generate_landmarks.py','save_landmarks'),('preprocessing/generate_diffs.py','save_diffs')]:
        p=args.source_root/path
        assert hashlib.sha256(p.read_bytes()).hexdigest()==next(r['sha256'] for r in pins['files'] if r['path']==path)
        nodes.append(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
    rng=np.random.default_rng(752)
    keys=[(0,0),(10,1),(20,2),(310,0),(320,0),(15,0),(30,0)]
    originals=[{'frame':f,'actor':a,'image':rng.integers(0,256,(48,48,3),dtype=np.uint8)} for f,a in keys]
    altered=[{'frame':r['frame'],'actor':r['actor'],'image':255-r['image']} for r in originals if r['frame']!=30]
    fs={}
    for identifier,records in [('synthetic_original',originals),('synthetic_altered',altered)]:
        for r in records:fs[f"crops/{identifier}/{r['frame']}_{r['actor']}.png"]=r['image'][:,:,::-1].copy()
    saved_landmarks={};saved_diffs={}
    def save_diff(path,values):
        parts=Path(path).stem.split('_');saved_diffs[(int(parts[0]),int(parts[1]))]=values.copy();return True
    namespace={'np':NumpyProxy(saved_landmarks),'Image':Image,
        'os':SimpleNamespace(path=SimpleNamespace(join=os.path.join,exists=lambda p:p in fs),makedirs=lambda *a,**kw:None),
        'cv2':SimpleNamespace(imread=lambda p,flag:fs[p].copy(),IMREAD_COLOR=cv2.IMREAD_COLOR,
                             cvtColor=cv2.cvtColor,COLOR_BGR2GRAY=cv2.COLOR_BGR2GRAY,imwrite=save_diff),
        'compare_ssim':lambda a,b,**kw:structural_similarity(a,b,channel_axis=-1,full=True)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-annotation-writers>','exec'),namespace)
    values=np.array([[[2.5,3.5],[4.5,5.5],[6.1,7.9],[8.,9.],[10.,11.]],
                     [[20.,21.],[22.,23.],[24.,25.],[26.,27.],[28.,29.]]],dtype=np.float32)
    comparisons=0
    for points in (values,None):
        saved_landmarks.clear();saved_diffs.clear();detector=Detector(points);namespace['detector']=detector
        namespace['save_landmarks']('synthetic_original.mp4','')
        namespace['save_diffs'](('synthetic_original','synthetic_altered'),'')
        actual=annotate_pair(originals,altered,Detector(points))
        assert set(actual['landmarks'])==set(saved_landmarks)
        assert set(actual['differences'])==set(saved_diffs)=={(0,0),(10,1),(310,0)}
        for key in saved_landmarks:np.testing.assert_array_equal(actual['landmarks'][key],saved_landmarks[key])
        for key in saved_diffs:np.testing.assert_array_equal(actual['differences'][key],saved_diffs[key])
        if points is not None:
            expected=np.array([[2,4],[4,6],[6,8],[8,9],[10,11]],dtype=np.int16)
            assert set(actual['landmarks'])=={(0,0),(10,1),(310,0),(30,0)}
            np.testing.assert_array_equal(actual['landmarks'][(0,0)],expected)
        comparisons+=1
    torch.set_num_threads(2)
    detector=build_preprocessing_detector(stage='landmarks',initialization='random',seed=753)
    with torch.no_grad():
        for p in detector.parameters():p.zero_()
        for model,head in ((detector.pnet,'conv4_1'),(detector.rnet,'dense5_1'),(detector.onet,'dense6_1')):
            getattr(model,head).bias.copy_(torch.tensor([-2.,2.]))
        detector.onet.dense6_3.bias.fill_(.5)
    image=originals[0]['image']
    _,_,points=detector.detect(Image.fromarray(image),landmarks=True)
    assert points.dtype==object
    try:np.around(points[0])
    except TypeError:pass
    else:raise AssertionError('expected modern object-array incompatibility not observed')
    actual=annotate_pair(originals[:1],altered[:1],detector)
    expected=np.around(np.asarray(points[0],dtype=np.float64)).astype(np.int16)
    np.testing.assert_array_equal(actual['landmarks'][(0,0)],expected)
    assert actual['landmarks'][(0,0)].shape==(5,2)
    files=['sciona/dfdc_crop_annotations.py','sciona/dfdc_difference_mask.py','sciona/dfdc_training_crops.py',
           'sciona/dfdc_detector.py','sciona/dfdc_mtcnn_networks.py','sciona/dfdc_mtcnn_cascade.py',
           'scripts/validate_dfdc_crop_annotations.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-crop-annotation-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'combined_source_writer_cases':comparisons,'independent_frame_actor_caps':True,
                 'independent_ties_to_even_rounding':True,'actual_landmark_detector_executed':True,
                 'current_object_array_source_failure_reproduced':True,'numeric_conversion_adaptation_verified':True},
       'adaptations':['CurrentMTCNN numeric object arrays converted to float64 before source rounding/int16cast; original source fails and omits annotations without this conversion.',
                      'Explicit sparse positional maps replace source media files; no original identities or logs.'],
       'limits':'Synthetic crops and constructed detector states only. Source writer comparison uses modernSSIM compatibility and previouslyvalidated difference-mask math. No pretrained landmark quality, full paired lifecycle, groupedfold or promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_crop_annotations.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

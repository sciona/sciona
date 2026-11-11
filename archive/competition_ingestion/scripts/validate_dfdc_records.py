"""Synthetic pair inventory, annotation reuse and actual detector assembly."""
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_records import build_records
from sciona.dfdc_training_crops import build_preprocessing_detector


class Boxes:
    def __init__(self):self.batch_sizes=[]
    def detect(self,images,*,landmarks):
        assert not landmarks;self.batch_sizes.append(len(images))
        boxes=np.array([[4.,4.,12.,12.],[12.,12.,22.,22.],[2.,2.,10.,10.]])
        return np.stack([boxes.copy() for _ in images]),None


class Landmarks:
    def __init__(self):self.calls=0
    def detect(self,image,*,landmarks):
        assert landmarks;self.calls+=1
        return None,None,np.array([[[3.,3.],[7.,3.],[5.,5.],[3.,7.],[7.,7.]]])


def main():
    definitions=[(0,0),(0,2),(2,3),(2,5),(2,3),(5,9)]
    clips=[{'original':original,'part':part,'frames':np.full((11,64,64,3),40+i*25,dtype=np.uint8)}
           for i,(original,part) in enumerate(definitions)]
    boxes,landmarks=Boxes(),Landmarks();before=random.getstate()
    records=build_records(clips,box_detector=boxes,landmark_detector=landmarks,record_order_seed=777)
    assert random.getstate()==before
    assert boxes.batch_sizes==[11,11] and landmarks.calls==8
    assert len(records)==20
    assert {r['clip_position'] for r in records}=={0,1,2,3,4}
    assert sum(r['label']==0 for r in records)==8 and sum(r['label']==1 for r in records)==12
    expected={(clip,frame,actor) for clip in range(5) for frame in (0,10) for actor in (0,1)}
    assert {(r['clip_position'],r['frame'],r['actor']) for r in records}==expected
    by_key={(r['clip_position'],r['frame'],r['actor']):r for r in records}
    for row in records:
        parent=by_key[(row['original_position'],row['frame'],row['actor'])]
        assert row['fold']==parent['fold']
        assert row['landmarks'] is parent['landmarks']
        assert np.all(row['image']==40+row['clip_position']*25)
        if row['label']==0:assert row['mask'] is None
        else:assert row['mask'] is not None and row['mask'].shape==row['image'].shape[:2]
    repeated=build_records(clips,box_detector=Boxes(),landmark_detector=Landmarks(),record_order_seed=777)
    order=lambda rows:[(r['clip_position'],r['frame'],r['actor']) for r in rows]
    assert order(records)==order(repeated)
    changed=build_records(clips,box_detector=Boxes(),landmark_detector=Landmarks(),record_order_seed=778)
    assert order(changed)!=order(records) and set(order(changed))==set(order(records))
    # Actual native MTCNN stage-specific networks, with synthetic states.
    torch.set_num_threads(2)
    detectors=[]
    for stage in ('boxes','landmarks'):
        detector=build_preprocessing_detector(stage=stage,initialization='random',seed=327)
        with torch.no_grad():
            for p in detector.parameters():p.zero_()
            for model,head in ((detector.pnet,'conv4_1'),(detector.rnet,'dense5_1'),(detector.onet,'dense6_1')):
                getattr(model,head).bias.copy_(torch.tensor([-2.,2.]))
            detector.onet.dense6_3.bias.fill_(.5)
        detectors.append(detector)
    original=np.random.default_rng(328).integers(1,256,(11,80,80,3),dtype=np.uint8)
    actual=build_records([{'original':0,'part':3,'frames':original},
                          {'original':0,'part':3,'frames':255-original}],
                         box_detector=detectors[0],landmark_detector=detectors[1],record_order_seed=329)
    assert len(actual)>0 and {r['label'] for r in actual}=={0,1}
    assert {r['frame'] for r in actual}=={0,10}
    assert all(r['landmarks'] is not None and r['landmarks'].shape==(5,2) for r in actual)
    assert all(r['mask'] is not None for r in actual if r['label']==1)
    files=['sciona/dfdc_records.py','sciona/dfdc_folds.py','sciona/dfdc_training_crops.py',
           'sciona/dfdc_crop_annotations.py','sciona/dfdc_difference_mask.py','sciona/dfdc_detector.py',
           'sciona/dfdc_faces.py','sciona/dfdc_mtcnn_networks.py','sciona/dfdc_mtcnn_cascade.py',
           'scripts/validate_dfdc_records.py']
    report={'format':'dfdc-record-assembly-validation.v1','result':'passed',
        'checks':{'synthetic_pair_records':20,'original_detection_batches':2,'original_landmark_calls':8,
                  'unpaired_original_excluded':True,'annotation_reuse_verified':True,
                  'explicit_shuffle_repeatability':True,'caller_python_rng_preserved':True,
                  'actual_detector_annotation_records':len(actual)},
        'adaptations':['Caller clip order and explicit localshuffle seed replace nondeterministic source set/process collection and unseededshuffle.',
                       'Positional references replace source mediaidentities; arrays are runtime-only.'],
        'limits':'Synthetic inventory expectations and actual constructed-state MTCNN assembly, relying on separately source-validated component gates. No original media, pretrainedquality, full DataLoader/training lifecycle, checkpointselection or promotion claim.',
        'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_records.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

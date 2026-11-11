"""Selected-state boundary order, precision and rejection checks with synthetic states."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_ensemble_prediction import predict_selected,build_selected_models
from sciona.dfdc_selection import selection_plan


class Model(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.bias=torch.nn.Parameter(torch.tensor([value],dtype=torch.float32))
    def forward(self,x):return self.bias.expand(len(x),1)


class Detector:
    def detect(self,image,landmarks=False):
        return np.array([[1.,1.,6.,6.]]),np.array([1.])


def main():
    plan=selection_plan([(s,0,2) for s in (111,555,777,888,999)],
                        [(111,0,0),(555,0,0),(777,0,0),(777,0,1),(888,0,0),(888,0,1),(999,0,0)])
    values=[-2.,-1.,-.5,0.,.5,1.,3.]
    snapshots=[{'kind':str(key[2]),'epoch':key[2]+1,'bce_best':100.,
                'state_dict':{'bias':torch.tensor([value])}} for key,value in zip(plan['requests'],values)]
    bundle={'plan':plan,'snapshots':snapshots}
    calls=[]
    def builder(*,initialization,seed,state):
        assert initialization=='state';calls.append((seed,float(state['bias'].item())))
        model=Model(0.);model.load_state_dict(state,strict=True);return model
    cases=0
    with patch('sciona.dfdc_ensemble_prediction.build_classifier',side_effect=builder):
        for precision in ('float32','float16'):
            calls.clear()
            outputs=predict_selected([np.full((2,19,23,3),90,dtype=np.uint8),
                                      np.empty((0,19,23,3),dtype=np.uint8)],bundle,
                                      detector=Detector(),frames_per_video=2,precision=precision)
            assert calls==[(row[0],value) for row,value in zip(plan['requests'],values)]
            probabilities=torch.sigmoid(torch.tensor(values,dtype=getattr(torch,precision))).numpy()
            assert outputs[0]['score']==float(np.mean(probabilities))
            assert outputs[0]['models']==7 and outputs[0]['classified_faces']==2 and outputs[0]['status']=='predicted'
            assert outputs[1]['score']==.5 and outputs[1]['reason']=='no_frames'
            cases+=1
        broken=copy.deepcopy(bundle);broken['snapshots'][0]['state_dict']['bias'].fill_(1e6)
        try:build_selected_models(broken,precision='float16')
        except ValueError:pass
        else:raise AssertionError('half overflow accepted')
    rejected=0
    for broken,precision in (({},'float32'),(dict(bundle,snapshots=snapshots[:-1]),'float32'),
                             (dict(bundle,plan=dict(plan,source_requests_preserved=True)),'float32'),
                             (bundle,'float64')):
        with patch('sciona.dfdc_ensemble_prediction.build_classifier',side_effect=AssertionError('loaded before validation')):
            try:build_selected_models(broken,precision=precision)
            except ValueError:rejected+=1
            else:raise AssertionError('invalid selection boundary accepted')
    paths=['sciona/dfdc_ensemble_prediction.py','sciona/dfdc_selection.py','sciona/dfdc_prediction.py',
           'scripts/validate_dfdc_ensemble_prediction.py']
    report={'format':'dfdc-ensemble-prediction-validation.v1','result':'passed',
            'checks':{'seven_model_order_precision_cases':cases,'load_once_for_two_clips':True,
                      'independent_mean_sigmoid_oracle':True,'half_conversion_overflow_rejected':True,
                      'invalid_metadata_before_model_load':rejected},
            'limits':'Synthetic constant-logit models isolate state-loading and order/precision boundary. Full trained B7 ensemble integration is separate; no checkpoint identity or pretrained quality claim.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_dfdc_ensemble_prediction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

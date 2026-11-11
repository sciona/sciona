"""Synthetic runtime JSON schema and intercepted full stage wiring checks."""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_codec import encode_array,encode_state
from sciona.dfdc_runtime import prepare,execute
from validate_dfdc_training import assert_equal


def payload():
    image=encode_array(np.zeros((2,19,23,3),dtype=np.uint8))
    return {'version':1,'training':[{'frames':image,'part':part,'original':original}
            for part,original in ((0,0),(0,0),(25,2),(25,2))],
            'prediction':[image],'detector':{'policy':'random','seed':17},
            'classifier':{'policy':'random'},'hull':{'predictor_path':'/synthetic/native-predictor'},
            'plan':{'runs':[[111,0,1]],'requests':[[111,0,0]]},
            'config':{'record_order_seed':18,'n_splits':2,'batch_size':1,'batches_per_epoch':1}}


def main():
    data=json.loads(json.dumps(payload()));ready=prepare(data)
    assert ready['config']['frames_per_video']==32 and ready['config']['precision']=='float16'
    assert np.array_equal(ready['training'][0]['frames'],np.zeros((2,19,23,3),dtype=np.uint8))
    assert ready['plan']['source_requests_preserved'] is False
    # Initialization states decoded before model construction; structural tensor
    # compatibility is independently enforced by the actual component factories.
    initialized=copy.deepcopy(data)
    state=encode_state({'synthetic':torch.ones(2)})
    initialized['detector']={'policy':'state','seed':17,'state':state}
    initialized['classifier']={'policy':'encoder','states':[{'seed':111,'fold':0,'state':state}]}
    decoded=prepare(initialized)
    assert torch.equal(decoded['detector']['state']['synthetic'],torch.ones(2))
    assert torch.equal(decoded['classifier']['states'][(111,0)]['synthetic'],torch.ones(2))
    calls=[];records=[{},{}]
    def hull(**kwargs):
        calls.append('hull');assert kwargs=={'predictor_path':data['hull']['predictor_path']}
        return 'hull-detector','hull-predictor'
    def preprocessing(**kwargs):
        calls.append(kwargs['stage']);assert kwargs['initialization']=='random' and kwargs['seed']==17
        return kwargs['stage']
    def record_builder(clips,**kwargs):
        calls.append('records');assert len(clips)==4
        assert kwargs==dict(box_detector='boxes',landmark_detector='landmarks',record_order_seed=18,n_splits=2)
        return records
    def training(received,plan,**kwargs):
        calls.append('training');assert received is records and plan==ready['plan']
        assert kwargs==dict(detector='hull-detector',predictor='hull-predictor',initialization='random',
                            states=None,batch_size=1,batches_per_epoch=1,test_every=1)
        random.random();np.random.random();torch.rand(1)
        return {'plan':plan,'runs':[{'synthetic':True}],'snapshots':[]}
    def detector(**kwargs):calls.append('inference-detector');return 'inference'
    def prediction(clips,bundle,**kwargs):
        calls.append('prediction');assert len(clips)==1 and bundle['plan']==ready['plan']
        assert kwargs==dict(detector='inference',frames_per_video=32,precision='float16')
        return [{'score':.5,'status':'source_fallback'}]
    state_before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    with ExitStack() as stack:
        for name,fn in [('load_hull_backend',hull),('build_preprocessing_detector',preprocessing),
                        ('build_records',record_builder),('train_ensemble',training),
                        ('build_detector',detector),('predict_selected',prediction)]:
            stack.enter_context(patch('sciona.dfdc_runtime.'+name,side_effect=fn))
        result=execute(data)
    assert calls==['hull','boxes','landmarks','records','training','inference-detector','prediction']
    assert_equal(state_before,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
    assert result['training_records']==2 and 'predictor_path' not in json.dumps(result)
    assert data['hull']['predictor_path'] not in json.dumps(result)
    rejected=0
    bad_cases=[]
    for field,value in [('version',True),('prediction',[]),('training',[]),('classifier',{'policy':'random','states':[]}),
                        ('detector',{'policy':'random','seed':True}),('hull',{'predictor_path':''}),
                        ('plan',{'runs':[[111,0,1]],'requests':[[111,0,1]]}),
                        ('config',{'record_order_seed':18,'precision':'float64'})]:
        bad=copy.deepcopy(data);bad[field]=value;bad_cases.append(bad)
    bad=copy.deepcopy(data);bad['training'][3]['original']=3;bad_cases.append(bad)
    bad=copy.deepcopy(data);bad['training'][0]['frames']=encode_array(np.zeros((2,19,23,3),dtype=np.float32));bad_cases.append(bad)
    bad=copy.deepcopy(data);bad['extra']=True;bad_cases.append(bad)
    for bad in bad_cases:
        with patch('sciona.dfdc_runtime.load_hull_backend',side_effect=AssertionError('native load before schema validation')):
            try:execute(bad)
            except ValueError:rejected+=1
            else:raise AssertionError('malformed runtime payload accepted')
    files=['sciona/dfdc_runtime.py','sciona/dfdc_codec.py','sciona/dfdc_selection.py',
           'sciona/dfdc_folds.py','scripts/validate_dfdc_runtime.py','scripts/validate_dfdc_training.py']
    report={'format':'dfdc-runtime-validation.v1','result':'passed',
            'checks':{'synthetic_json_decoding':True,'explicit_state_decoding':True,
                      'complete_stage_wiring_intercepted':True,'caller_rng_restored':True,
                      'private_path_excluded_from_results':True,'pre_execution_rejections':rejected},
            'limits':'Runtime schema and stage wiring only; expensive/native stages intercepted. Full actual detector/dlib/B7 runtime and serialized graph evidence remain required.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

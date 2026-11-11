"""Full JSON DFDC runtime with synthetic media and actual native/neural stages."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_codec import encode_array,encode_state
from sciona.dfdc_detector import build_detector
from sciona.dfdc_runtime import execute
from validate_dfdc_hull_backend import train_synthetic
from validate_dfdc_training import assert_equal


def main():
    torch.set_num_threads(2)
    detector=build_detector(initialization='random',seed=327)
    with torch.no_grad():
        for parameter in detector.parameters():parameter.zero_()
        for model,head in ((detector.pnet,'conv4_1'),(detector.rnet,'dense5_1'),(detector.onet,'dense6_1')):
            getattr(model,head).bias.copy_(torch.tensor([-2.,2.]))
        detector.onet.dense6_3.bias.fill_(.5)
    state=encode_state(detector.state_dict());del detector
    original=np.random.default_rng(328).integers(1,256,(11,80,80,3),dtype=np.uint8)
    frames=[original,255-original,original.copy(),255-original]
    training=[{'frames':encode_array(frame),'part':part,'original':parent}
              for frame,part,parent in zip(frames,(0,0,25,25),(0,0,2,2))]
    predictor,_,_,_=train_synthetic()
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-runtime-') as directory:
        native=Path(directory)/'predictor.dat';predictor.save(str(native))
        payload={'version':1,'training':training,'prediction':[encode_array(original[:2])],
                 'detector':{'policy':'state','seed':327,'state':state},
                 'classifier':{'policy':'random'},'hull':{'predictor_path':str(native)},
                 'plan':{'runs':[[seed,0,2] for seed in (111,555,777,888,999)],
                         'requests':[[111,0,0],[555,0,0],[777,0,0],[777,0,1],[888,0,0],[888,0,1],[999,0,0]]},
                 'config':{'record_order_seed':329,'n_splits':2,'batch_size':1,'batches_per_epoch':1,
                           'frames_per_video':2,'precision':'float32'}}
        payload=json.loads(json.dumps(payload,allow_nan=False))
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        with warnings.catch_warnings(),patch('torch.load',side_effect=AssertionError('implicit pickle load')):
            warnings.simplefilter('ignore',UserWarning)
            result=execute(payload)
        assert_equal(before,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
        assert len(result['runs'])==5 and result['training_records']>0
        assert sum(e['training']['optimizer_updates'] for run in result['runs'] for e in run['training']['epochs'])==10
        assert all(e['validation']['clips']==2 for run in result['runs'] for e in run['training']['epochs'])
        prediction=result['predictions'][0]
        assert prediction['models']==7 and prediction['status']=='predicted'
        assert prediction['classified_faces']>=2 and np.isfinite(prediction['score'])
        assert not result['plan']['source_requests_preserved'] and not result['plan']['source_runs_preserved']
        serialized=json.dumps(result,allow_nan=False)
        assert str(native) not in serialized and 'predictor_path' not in serialized and 'state_dict' not in serialized
    files=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('dfdc_*.py'))]
    files+=['scripts/validate_dfdc_runtime_full.py','scripts/validate_dfdc_hull_backend.py','scripts/validate_dfdc_training.py']
    report={'format':'dfdc-runtime-full-validation.v1','result':'passed',
            'checks':{'json_input_roundtrip':True,'actual_MTCNN_preparation_and_prediction':True,
                      'actual_native_synthetic_predictor_loaded':True,'full_B7_training_runs':5,
                      'connected_training_updates':10,'validation_epochs':10,'selected_ensemble_models':7,
                      'nonfallback_prediction':True,'caller_rng_restored':True,'runtime_path_and_states_excluded_from_output':True},
            'aggregate':{'training_records':result['training_records'],'classified_faces':prediction['classified_faces']},
            'limits':'All runtime stages executed without stand-ins. Detector uses constructed synthetic state; dlib68-point predictor trained only on synthetic images. Native loader validates predictor but positive HOG face quality is not claimed. Five B7 runs use random initialization, two epochs/one batch1 update each,380 CPUfloat32. Explicit shortened request plan, not original checkpoint provenance or default-scale quality. Graph/provider execution remains separate.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_runtime_full.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

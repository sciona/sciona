"""Actual five-run B7 ensemble training on an explicit shortened synthetic plan."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_ensemble_training import train_ensemble
from sciona.dfdc_selection import selection_plan
from sciona.dfdc_classifier import build_classifier
from sciona.dfdc_ensemble_prediction import predict_selected
from validate_dfdc_ensemble_prediction import Detector


def main():
    torch.set_num_threads(2)
    plan=selection_plan([(s,0,2) for s in (111,555,777,888,999)],
                        [(111,0,0),(555,0,0),(777,0,0),(777,0,1),(888,0,0),(888,0,1),(999,0,0)])
    rows=[]
    for fold in (0,1):
        for label in (0,1):
            pixels=np.arange(19*23*3,dtype=np.uint16).reshape(19,23,3)
            rows.append({'image':((pixels+label*67)%256).astype(np.uint8),'label':label,
                         'fold':fold,'frame':0,'clip_position':fold*2+label,'mask':None,'landmarks':None})
    with warnings.catch_warnings(),patch('torch.load',side_effect=AssertionError('implicit weight load')):
        warnings.simplefilter('ignore',UserWarning)
        result=train_ensemble(rows,plan,detector=lambda *a:[],predictor=lambda *a:None,
                              initialization='random',batch_size=1,batches_per_epoch=1)
        assert len(result['runs'])==5 and len(result['snapshots'])==7
        assert sum(e['training']['optimizer_updates'] for r in result['runs'] for e in r['training']['epochs'])==10
        assert all(e['validation']['clips']==2 for r in result['runs'] for e in r['training']['epochs'])
        expected=[1,1,1,2,1,2,1]
        assert [s['epoch'] for s in result['snapshots']]==expected
        model=build_classifier(initialization='random',seed=920).eval()
        # Reuse one model to check every selected full state without retaining seven models.
        expected_state=model.state_dict()
        with torch.no_grad():
            for snapshot in result['snapshots']:
                state=snapshot['state_dict']
                assert state.keys()==expected_state.keys()
                assert all(state[k].shape==v.shape and state[k].dtype==v.dtype and torch.isfinite(state[k]).all()
                           for k,v in expected_state.items())
                model.load_state_dict(state,strict=True)
                output=model(torch.zeros(1,3,380,380))
                assert output.shape==(1,1) and torch.isfinite(output).all()
        del model, expected_state
        captured=[]
        def tracked_builder(**kwargs):
            model=build_classifier(**kwargs)
            model.register_forward_hook(lambda module,inputs,output: captured.append(torch.sigmoid(output.detach()).cpu().numpy()))
            return model
        clips=[np.full((2,19,23,3),90,dtype=np.uint8)]
        with patch('sciona.dfdc_ensemble_prediction.build_classifier',side_effect=tracked_builder):
            predictions=predict_selected(clips,result,detector=Detector(),frames_per_video=2,precision='float32')
        assert len(captured)==7 and all(p.shape==(2,1) for p in captured)
        assert all(np.array_equal(p[0],p[1]) for p in captured)
        assert predictions[0]['status']=='predicted' and predictions[0]['models']==7
        assert predictions[0]['score']==float(np.mean([p.mean() for p in captured]))
        assert not torch.equal(result['snapshots'][2]['state_dict']['fc.weight'],
                               result['snapshots'][3]['state_dict']['fc.weight'])
    files=['sciona/dfdc_ensemble_training.py','sciona/dfdc_training.py','sciona/dfdc_selection.py',
           'sciona/dfdc_classifier.py','sciona/dfdc_stochastic_depth.py','sciona/dfdc_ensemble_prediction.py','sciona/dfdc_prediction.py',
           'scripts/validate_dfdc_ensemble_prediction.py','scripts/validate_dfdc_ensemble_b7.py']
    report={'format':'dfdc-ensemble-b7-validation.v1','result':'passed',
            'checks':{'full_B7_training_runs':5,'connected_updates':10,'validation_epochs':10,
                      'selected_full_states':7,'strict_state_load_and_finite_forward_checks':7,
                      'repeated_seed_distinct_epoch_states':True,'trained_seven_state_ensemble_prediction':True,
                      'independent_identical_face_mean_oracle':True},
            'limits':'Explicit shortened synthetic plan, two epochs per seed, one batch1 update per epoch at380 CPUfloat32, random initialization. No-face training hull stand-in and controlled inference boxes; full historical training length, pretrained quality and original seven checkpoint identities not claimed.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_ensemble_b7.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

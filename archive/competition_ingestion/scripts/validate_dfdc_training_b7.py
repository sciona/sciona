"""Actual full B7 connected training/validation/checkpoints on synthetic records."""
import hashlib
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_classifier import build_classifier
from sciona.dfdc_scheduler import PolyLR
from sciona.dfdc_training import train


def main():
    torch.set_num_threads(2)
    rows=[]
    for fold in (0,1):
        for label in (0,1):
            pixels=np.arange(23*29*3,dtype=np.uint16).reshape(23,29,3)
            rows.append({'image':((pixels+label*71)%256).astype(np.uint8),
                         'label':label,'fold':fold,'frame':0,'clip_position':fold*2+label,
                         'mask':None,'landmarks':None})
    events=[]
    def sink(event):
        assert event['state_dict'] and all(torch.isfinite(v).all() for v in event['state_dict'].values())
        events.append({k:v for k,v in event.items() if k!='state_dict'})
    with warnings.catch_warnings(), patch('torch.load',side_effect=AssertionError('implicit state load')):
        warnings.simplefilter('ignore',UserWarning)
        model=build_classifier(initialization='random',seed=611)
        before=model.fc.bias.detach().clone()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0001,nesterov=True)
        scheduler=PolyLR(optimizer,max_iter=100500)
        state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        result=train(model,optimizer,scheduler,rows,detector=lambda *a:[],predictor=lambda *a:None,
                     seed=111,checkpoint_sink=sink,epochs=1,batch_size=1,batches_per_epoch=2)
        assert random.getstate()==state[0]
        current=np.random.get_state()
        assert current[0]==state[1][0] and np.array_equal(current[1],state[1][1]) and current[2:]==state[1][2:]
        assert torch.equal(torch.get_rng_state(),state[2])
        assert not torch.equal(before,model.fc.bias)
        epoch=result['epochs'][0]
        assert epoch['training']['optimizer_updates']==2 and epoch['training']['examples']==2
        assert epoch['validation']['clips']==2 and epoch['validation']['samples']==2
        assert np.isfinite(epoch['validation']['loss'])
        assert [e['kind'] for e in events]==['last','0','best_dice','last']
        assert [e['bce_best'] for e in events][:2]==[100.,100.]
        assert events[2]['bce_best']==events[3]['bce_best']==result['best_loss']
        assert model.encoder.classifier.weight.grad is None
    files=['sciona/dfdc_training.py','sciona/dfdc_classifier.py','sciona/dfdc_stochastic_depth.py','docs/reviews/competition_dfdc_stochastic_depth.json','sciona/dfdc_dataset.py',
           'sciona/dfdc_preparation.py','sciona/dfdc_transforms.py','sciona/dfdc_epoch.py',
           'sciona/dfdc_loss.py','sciona/dfdc_scheduler.py','sciona/dfdc_evaluation.py',
           'sciona/dfdc_checkpoints.py','scripts/validate_dfdc_training_b7.py']
    report={'format':'dfdc-training-b7-validation.v1','result':'passed',
            'checks':{'actual_full_B7_training_updates':2,'actual_full_B7_validation_clips':2,
                      'source_resolution':380,'checkpoint_events':4,'head_updated':True,
                      'caller_rng_restored':True,'implicit_state_load_forbidden':True},
            'aggregate_validation_loss':result['best_loss'],
            'limits':'One synthetic epoch, random B7 using installed timm, CPU float32 batch1 workers0. Actual augmentation, validation and checkpoint lifecycle. Hull branch has no-face detector stand-in; native dlib evidence is separate. No 40-epoch/2500-batch quality, historical weights or seven-checkpoint provenance claim.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_training_b7.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

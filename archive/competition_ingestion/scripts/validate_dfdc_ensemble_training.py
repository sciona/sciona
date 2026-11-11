"""Five independent runs and seven snapshots through actual preparation/training."""
import hashlib
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_ensemble_training import _train_runs, train_ensemble
from sciona.dfdc_selection import selection_plan
from sciona.dfdc_scheduler import PolyLR
from sciona.dfdc_training import train
from validate_dfdc_training import Model, assert_equal


def main():
    torch.set_num_threads(2)
    seeds=(111,555,777,888,999)
    # Explicit shortened synthetic plan; original seven checkpoint epochs are not relabeled.
    plan=selection_plan([(s,0,2) for s in seeds],[(111,0,0),(555,0,0),(777,0,0),
                        (777,0,1),(888,0,0),(888,0,1),(999,0,0)])
    assert not plan['source_requests_preserved'] and not plan['source_runs_preserved']
    rows=[]
    for fold in (0,1):
        for label in (0,1):
            pixels=np.arange(19*23*3,dtype=np.uint16).reshape(19,23,3)
            rows.append({'image':((pixels+label*67)%256).astype(np.uint8),'label':label,
                         'fold':fold,'frame':0,'clip_position':fold*2+label,'mask':None,'landmarks':None})
    calls=[]
    def factory(seed,fold):
        calls.append((seed,fold))
        torch.manual_seed(seed)
        model=Model()
        opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0001,nesterov=True)
        return model,opt,PolyLR(opt,max_iter=100500)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        actual=_train_runs(rows,plan,factory,detector=lambda *a:[],predictor=lambda *a:None,
                           batch_size=1,batches_per_epoch=1,test_every=1)
        assert_equal(state,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
        assert calls==[(s,0) for s in seeds]
        expected={};metrics=[]
        for seed in seeds:
            model,opt,sched=factory(seed,0)
            def collect(event):
                if event['kind'].isdecimal():expected[(seed,0,int(event['kind']))]=event
            metrics.append(train(model,opt,sched,rows,detector=lambda *a:[],predictor=lambda *a:None,
                                 seed=seed,checkpoint_sink=collect,epochs=2,batch_size=1,batches_per_epoch=1))
        assert_equal(actual['snapshots'],[expected[tuple(row)] for row in plan['requests']])
        assert_equal([row['training'] for row in actual['runs']],metrics)
        assert len(actual['snapshots'])==7
        # Failure during a later run must not leak factory RNG changes or return partial success.
        state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        def failing_factory(seed,fold):
            random.random();np.random.random();torch.rand(1)
            if seed==555:raise RuntimeError('synthetic factory failure')
            return factory(seed,fold)
        try:_train_runs(rows,plan,failing_factory,detector=lambda *a:[],predictor=lambda *a:None,
                        batch_size=1,batches_per_epoch=1,test_every=1)
        except RuntimeError as error:assert str(error)=='synthetic factory failure'
        else:raise AssertionError('partial failed execution returned success')
        assert_equal(state,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
        for mode,states in [('unknown',None),('random',{}),('encoder',{})]:
            try:train_ensemble(rows,plan,detector=None,predictor=None,initialization=mode,states=states)
            except ValueError:pass
            else:raise AssertionError('invalid initialization accepted')
    files=['sciona/dfdc_ensemble_training.py','sciona/dfdc_training.py','sciona/dfdc_selection.py',
           'sciona/dfdc_checkpoints.py','scripts/validate_dfdc_ensemble_training.py','scripts/validate_dfdc_training.py']
    report={'format':'dfdc-ensemble-training-validation.v1','result':'passed',
            'checks':{'independent_runs':5,'selected_snapshots':7,'connected_training_updates':10,
                      'individual_run_snapshots_and_metrics_exact':True,'rng_restoration_success_and_failure':True,
                      'initialization_rejections':3},
            'limits':'Actual synthetic preparation, augmentation, training/validation and checkpoints with small classifiers and explicit shortened epochs. Full B7 per-run evidence is separate; this does not prove default-scale five-run B7 execution or historical seven-state provenance.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_ensemble_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

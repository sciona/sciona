"""Exercise source training controllers and structured resume on synthetic data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sciona.champs_source_runtime import ChampsSourceRuntime
from sciona.champs_training import ChampsTrainer,TrainingOptions


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.25))
        self.dropout=torch.nn.Dropout(.2)

    def forward(self,atoms,positions,bonds,*rest):
        return self.weight*self.dropout(torch.ones(atoms.shape[0],68,bonds.shape[1])),None


def equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right): equal(a,b)
    else:
        assert left==right


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runtime=ChampsSourceRuntime(args.source)
    atoms=torch.ones(4,12,3,dtype=torch.long)
    positions=torch.zeros(4,12,5)
    positions[:,:,0]=torch.arange(12)
    bonds=torch.zeros(4,24,5,dtype=torch.long)
    bonds[:,:,0]=torch.arange(24)%8+1
    bonds[:,:,1]=torch.arange(24)+1
    bonds[:,:,3]=torch.arange(24)%12
    bonds[:,:,4]=(torch.arange(24)+1)%12
    targets=torch.zeros(4,24,4)
    targets[:,:,0]=10.
    targets[:,:,2]=2.
    targets[:,:,3]=1.
    batch=(torch.arange(4),atoms,positions,bonds,torch.ones(4,24),
           torch.zeros(4,1,7,dtype=torch.long),torch.zeros(4,1),
           torch.zeros(4,1,10,dtype=torch.long),torch.zeros(4,1),targets)
    original=[x.clone() for x in batch]
    results=[]
    for optimizer in ('SGD','Adam','Adagrad','RAdam'):
        for scheduler in ('constant','cosine','inv_sqrt','dev_perf'):
            options=TrainingOptions(optim=optimizer,scheduler=scheduler,lr=.01,warmup_step=1,
                max_step=8,batch_size=4,batch_chunk=2,cutout=.5,seed=1729)
            trainer=ChampsTrainer(runtime,'model_H',SyntheticModel(),options)
            caller_torch=torch.random.get_rng_state().clone()
            caller_numpy=np.random.get_state()
            trainer.epoch([batch]*2)
            trainer.validation_schedule_step(1.)
            payload=io.BytesIO()
            torch.save(trainer.state_dict(),payload)
            payload.seek(0)
            saved=torch.load(payload,weights_only=True)
            trainer.epoch([batch]*2)
            trainer.validation_schedule_step(2.)
            expected=trainer.state_dict()
            resumed=ChampsTrainer(runtime,'model_H',SyntheticModel(),options)
            resumed.load_state_dict(saved)
            resumed.epoch([batch]*2)
            resumed.validation_schedule_step(2.)
            equal(resumed.state_dict(),expected)
            assert torch.equal(caller_torch,torch.random.get_rng_state())
            nr=np.random.get_state()
            np.testing.assert_array_equal(caller_numpy[1],nr[1])
            assert caller_numpy[0]==nr[0] and caller_numpy[2:]==nr[2:]
            for a,b in zip(batch,original): torch.testing.assert_close(a,b,rtol=0,atol=0)
            results.append({'optimizer':optimizer,'scheduler':scheduler,'exact_resume':True,
                            'cutout_and_dropout_enabled':True,'caller_rng_and_inputs_preserved':True})
    repo=Path(__file__).resolve().parents[1]
    args.output.write_text(json.dumps({'source_commit':runtime.commit,'results':results,
        'scope':'Sixteen optimizer/scheduler controllers on synthetic analytic model, original cutout and stochastic dropout, structured weights-only state resume. Full graph-transformer epoch integration remains separate.',
        'source_behavior':'Inverse-square-root scheduler is initialized but original epoch does not step it after warmup; preserved, not silently repaired.',
        'runtime_sha256':hashlib.sha256((repo/'sciona/champs_training.py').read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('All 16 optimizer/scheduler combinations resume exactly with stochastic cutout/dropout')


if __name__=='__main__':
    main()

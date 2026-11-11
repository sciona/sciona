"""Synthetic Adam history oracles, clipping boundaries and serialized continuation."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from sciona.webtraffic_adam import initial_state,adam_step


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==pins['files']['model.py']
    counts=dict(updates=0,resumed_updates=0,clipping_boundaries=0,rejections=0)
    for seed in range(8):
        rng=np.random.RandomState(seed+382)
        params=[torch.tensor(rng.normal(size=shape),dtype=torch.float32) for shape in [(3,4),(5,)]]
        state=initial_state(params);reference=[p.numpy().copy() for p in params]
        histories=[[],[]];bp1=np.float32(.9);bp2=np.float32(.999)
        resumed=None
        for step in range(1,17):
            magnitude=[0.,1e-10,.1,100.][step%4]
            gradients=[torch.tensor(rng.normal(size=p.shape)*magnitude,dtype=torch.float32) for p in params]
            saved=[p.clone() for p in params];old_state=copy.deepcopy(state)
            output,new_state,norm=adam_step(params,gradients,state)
            # Closed-form moment sums over all preceding clipped gradients.
            gn=np.sqrt(sum(np.sum(g.numpy().astype(np.float64)**2) for g in gradients))
            factor=min(1.,10/gn) if gn else 1.
            for i,g in enumerate(gradients):histories[i].append(g.numpy().astype(np.float64)*factor)
            rate=np.float32(.001)*np.sqrt(1-bp2)/(1-bp1)
            for i,history in enumerate(histories):
                beta1=float(np.float32(.9));beta2=float(np.float32(.999))
                mean=sum((1-beta1)*beta1**(step-j-1)*g for j,g in enumerate(history))
                variance=sum((1-beta2)*beta2**(step-j-1)*g*g for j,g in enumerate(history))
                reference[i]=reference[i]-rate*mean/(np.sqrt(variance)+1e-8)
                np.testing.assert_allclose(output[i],reference[i],rtol=2e-6,atol=2e-7)
                np.testing.assert_allclose(new_state['m'][i],mean,rtol=2e-5,atol=2e-7)
                np.testing.assert_allclose(new_state['v'][i],variance,rtol=2e-6,atol=2e-8)
                torch.testing.assert_close(params[i],saved[i],rtol=0,atol=0)
                torch.testing.assert_close(state['m'][i],old_state['m'][i],rtol=0,atol=0)
            bp1*=np.float32(.9);bp2*=np.float32(.999)
            if resumed is not None:
                rp,rs,_=adam_step(resumed['parameters'],gradients,resumed['state'])
                for a,b in zip(rp,output):torch.testing.assert_close(a,b,rtol=0,atol=0)
                for key in ['m','v']:
                    for a,b in zip(rs[key],new_state[key]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                resumed=dict(parameters=rp,state=rs);counts['resumed_updates']+=1
            params,state=output,new_state;counts['updates']+=1
            if step==7:
                buffer=io.BytesIO();torch.save(dict(parameters=params,state=state),buffer);buffer.seek(0)
                resumed=torch.load(buffer,weights_only=True)
    for values in [[0.,0.],[6.,8.],[12.,16.]]:
        p=[torch.zeros(2)];g=[torch.tensor(values)];out,state,norm=adam_step(p,g,initial_state(p))
        expected=torch.tensor(values);expected=expected*min(1.,10/norm.item()) if norm else expected
        torch.testing.assert_close(state['m'][0],expected*(1-torch.tensor(.9)))
        counts['clipping_boundaries']+=1
    for bad in ['nan_gradient','missing_slot','negative_variance','bad_power']:
        p=[torch.ones(2)];g=[torch.ones(2)];s=initial_state(p)
        if bad=='nan_gradient':g[0][0]=float('nan')
        elif bad=='missing_slot':s['m']=[]
        elif bad=='negative_variance':s['v'][0][0]=-1
        else:s['beta1_power']=torch.tensor(1.)
        try:adam_step(p,g,s)
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Invalid optimizer input accepted')
    paths=['sciona/webtraffic_adam.py','scripts/validate_webtraffic_adam.py',
           'docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                equations_source='https://raw.githubusercontent.com/tensorflow/tensorflow/v1.10.0/tensorflow/python/training/adam.py',
                limitations=['Float32 dense gradients and explicit state; no TF kernel bitwise equivalence.',
                             'Continuation uses private in-memory synthetic serialization, not TF checkpoint format.',
                             'Optimizer step is separate from trainer shared global step; EMA ordering remains unresolved.',
                             'Full model optimizer/checkpoint lifecycle integration remains.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_adam.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

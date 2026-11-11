"""Synthetic EMA history oracles, model isolation and snapshot continuation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from sciona.webtraffic_ema import initial_ema,update_ema,ema_parameters


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==pins['files']['model.py']
    counts=dict(updates=0,resumed_updates=0,decay_boundaries=0,inference_copies=0,rejections=0)
    for seed in range(8):
        rng=np.random.RandomState(427+seed)
        models=[{'kernel':torch.tensor(rng.normal(size=(3,4)),dtype=torch.float32),
                 'bias':torch.tensor(rng.normal(size=4),dtype=torch.float32)} for _ in range(3)]
        shadows=[initial_ema(p) for p in models]
        initial=[{k:v.numpy().astype(np.float64).copy() for k,v in p.items()} for p in models]
        history=[[],[],[]];decays=[];resumed=None
        for iteration,step in enumerate([0,1,2,10,100,889,890,891,2000,10500,11500]):
            decay=np.minimum(np.float32(.99),(np.float32(1)+np.float32(step))/(np.float32(10)+np.float32(step)))
            decays.append(float(decay))
            snapshots=[{k:torch.tensor(rng.normal(size=v.shape),dtype=torch.float32) for k,v in p.items()} for p in models]
            new=[]
            for i,(shadow,snapshot) in enumerate(zip(shadows,snapshots)):
                before={k:v.clone() for k,v in shadow.items()}
                actual,d=update_ema(shadow,snapshot,step)
                np.testing.assert_allclose(d,decay,rtol=0,atol=0)
                history[i].append({k:v.numpy().astype(np.float64) for k,v in snapshot.items()})
                for name,a in actual.items():
                    expected=initial[i][name]*np.prod(decays)
                    for j,saved in enumerate(history[i]):expected=expected+saved[name]*(1-decays[j])*np.prod(decays[j+1:])
                    np.testing.assert_allclose(a,expected,atol=2e-7,rtol=3e-6)
                    torch.testing.assert_close(shadow[name],before[name],rtol=0,atol=0)
                new.append(actual);counts['updates']+=1
            if resumed is not None:
                restored=[]
                for i,snapshot in enumerate(snapshots):
                    a,_=update_ema(resumed[i],snapshot,step)
                    for name in a:torch.testing.assert_close(a[name],new[i][name],rtol=0,atol=0)
                    restored.append(a);counts['resumed_updates']+=1
                resumed=restored
            shadows=new
            if iteration==4:
                buffer=io.BytesIO();torch.save(shadows,buffer);buffer.seek(0);resumed=torch.load(buffer,weights_only=True)
        for shadow in shadows:
            prediction=ema_parameters(shadow,['kernel','bias'])
            prediction['bias'].add_(3.)
            assert not torch.equal(prediction['bias'],shadow['bias']);counts['inference_copies']+=1
    for step,expected in [(0,.1),(1,2/11),(889,890/899),(890,.99),(891,.99)]:
        _,d=update_ema({'x':torch.ones(1)},{'x':torch.zeros(1)},step)
        np.testing.assert_allclose(d,expected,rtol=1e-7);counts['decay_boundaries']+=1
    # Explicitly demonstrate why read ordering cannot be silently chosen.
    base={'x':torch.tensor([1.])};post={'x':torch.tensor([2.])}
    old_read,_=update_ema(base,base,0);new_read,_=update_ema(base,post,0)
    assert not torch.equal(old_read['x'],new_read['x'])
    later_step,_=update_ema(base,post,1)
    assert not torch.equal(new_read['x'],later_step['x'])
    for snapshot,step in [({},0),({'x':torch.ones(2)},0),({'x':torch.tensor([float('nan')])},0),(post,-1),(post,True)]:
        try:update_ema(base,snapshot,step)
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Invalid EMA input accepted')
    paths=['sciona/webtraffic_ema.py','scripts/validate_webtraffic_ema.py','docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                equations_source='https://raw.githubusercontent.com/tensorflow/tensorflow/v1.10.0/tensorflow/python/training/moving_averages.py',
                limitations=['Explicit snapshot and observed global step required; Adam/EMA/step scheduling is unresolved.',
                             'Source variable initialization and no zero-debias semantics preserved; float32 only.',
                             'Synthetic in-memory EMA restoration, not TensorFlow checkpoint format or full lifecycle.',
                             'Numerical oracles do not establish original TensorFlow runtime parity.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_ema.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

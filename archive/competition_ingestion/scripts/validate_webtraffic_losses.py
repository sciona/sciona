"""Synthetic float32 loss oracles and autograd finite differences; no TF runtime claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sciona.webtraffic_losses import calc_loss, decode_predictions, rnn_stability_loss, rnn_activation_loss


def oracle(pred, true, extra):
    valid=np.isfinite(true); true=np.where(valid,true,0); weights=valid.astype(np.float32)*extra[None,:]
    n=np.count_nonzero(weights)
    mae=np.sum(np.abs(true-pred)*weights)/n if n else 0.
    a,b=np.expm1(true),np.expm1(pred)
    smooth=np.sum(2*np.abs(a-b)/np.maximum(np.abs(a)+np.abs(b)+.1,.6)*weights)/n if n else 0.
    a,b=np.round(a),np.maximum(np.round(b),0)
    denom=np.abs(a)+np.abs(b)
    with np.errstate(invalid='ignore',divide='ignore'):
        rounded=np.where(denom<.01,0,2*np.abs(a-b)/denom)
        score=np.sum(rounded*weights)/np.sum(weights)
    return mae,smooth,score,true.size


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==pins['files']['model.py']
    counts=dict(loss_cases=0,gradient_checks=0,regularizer_checks=0,decode_checks=0)
    for seed in range(8):
        rng=np.random.RandomState(seed+711)
        pred=rng.uniform(.2,2,(3,7)).astype(np.float32)
        true=rng.uniform(.2,2,(3,7)).astype(np.float32);true[0,0]=np.nan;true[1,1]=np.inf
        for extra in [np.ones(7,dtype=np.float32),np.array([0,.2,.4,0,1,1,0],dtype=np.float32),np.zeros(7,dtype=np.float32)]:
            p=torch.tensor(pred,requires_grad=True)
            a=calc_loss(p,torch.tensor(true),torch.tensor(extra));b=oracle(pred,true,extra)
            np.testing.assert_allclose([x.detach().numpy() for x in a[:3]],b[:3],rtol=3e-6,atol=1e-7,equal_nan=True)
            assert a[3]==b[3];counts['loss_cases']+=1
            a[1].backward()
            assert p.grad[0,0]==0 and p.grad[1,1]==0
            for i,j in [(2,2),(2,4)]:
                step=.001
                plus=pred.copy();minus=pred.copy();plus[i,j]+=step;minus[i,j]-=step
                finite=(oracle(plus,true,extra)[1]-oracle(minus,true,extra)[1])/(2*step)
                np.testing.assert_allclose(p.grad[i,j].item(),finite,atol=6e-5,rtol=.01)
                counts['gradient_checks']+=1
        output=rng.normal(size=(5,3,4)).astype(np.float32)
        t=torch.tensor(output,requires_grad=True)
        for beta in [0.,.03]:
            lengths=np.linalg.norm(output,axis=-1)
            expected=beta*np.mean(np.diff(lengths,axis=0)**2)
            a=rnn_stability_loss(t,beta)
            np.testing.assert_allclose(a.detach().numpy() if torch.is_tensor(a) else a,expected,rtol=1e-6)
            a=rnn_activation_loss(t,beta)
            np.testing.assert_allclose(a.detach().numpy() if torch.is_tensor(a) else a,.5*beta*np.sum(output**2),rtol=1e-6)
            counts['regularizer_checks']+=2
        mean=rng.normal(size=3).astype(np.float32);std=rng.uniform(.2,2,size=3).astype(np.float32)
        readout=rng.normal(size=(7,3)).astype(np.float32)
        np.testing.assert_allclose(decode_predictions(torch.tensor(readout),torch.tensor(mean),torch.tensor(std)).numpy(),readout.T*std[:,None]+mean[:,None],rtol=1e-6)
        counts['decode_checks']+=1
    paths=['sciona/webtraffic_losses.py','scripts/validate_webtraffic_losses.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                semantics_source='https://raw.githubusercontent.com/tensorflow/tensorflow/v1.10.0/tensorflow/python/ops/losses/losses_impl.py',
                limitations=['NumPy formula oracles and PyTorch gradients, not original TensorFlow runtime parity.',
                             'Float32 finite predictions tested; masked nonfinite targets and fractional/zero masks covered.',
                             'All-zero mask gives zero reduced training losses but source rounded metric remains NaN.',
                             'Regularizer axis order is preserved as passed; full encoder/decoder layout integration remains.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_losses.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

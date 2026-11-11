"""Independent NumPy recurrence and finite-difference tests, synthetic weights only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sciona.webtraffic_decoder import decode, convert_encoder_state


def numpy_step(x,h,w):
    joined=np.concatenate([x,h],axis=1)
    a=joined.dot(w['gate_kernel'])+w['gate_bias']
    gates=1/(1+np.exp(-a));r,u=np.split(gates,2,axis=1)
    c=np.tanh(np.concatenate([x,h*r],axis=1).dot(w['candidate_kernel'])+w['candidate_bias'])
    return (1-u)*c+u*h


def reference(states,features,previous,cells,proj,bias,attention):
    targets=[];outputs=[]
    for day in range(features.shape[1]):
        arrays=[previous[:,None],features[:,day]]
        if attention is not None:arrays.append(attention[:,day])
        value=np.concatenate(arrays,axis=1);updated=[]
        for h,w in zip(states,cells):
            value=numpy_step(value,h,w);updated.append(value)
        states=updated;previous=(value.dot(proj)+bias).squeeze(-1)
        targets.append(previous);outputs.append(value)
    return np.stack(targets),np.stack(outputs),states


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==pins['files']['model.py']
    counts=dict(rollouts=0,state_mappings=0,gradient_checks=0)
    for seed in range(8):
        rng=np.random.RandomState(seed+713)
        for layers in [1,2]:
            for with_attention in [False,True]:
                batch,days,depth=2,7,4
                feat=rng.normal(size=(batch,days,3))*.2
                attn=rng.normal(size=(batch,days,2))*.2 if with_attention else None
                previous=rng.normal(size=batch)*.2
                states=rng.normal(size=(layers,batch,depth))*.2
                cells=[]
                for layer in range(layers):
                    ins=4+(2 if with_attention else 0) if layer==0 else depth
                    cells.append(dict(gate_kernel=rng.normal(size=(ins+depth,2*depth))*.2,
                                      gate_bias=np.ones(2*depth),candidate_kernel=rng.normal(size=(ins+depth,depth))*.2,
                                      candidate_bias=np.zeros(depth)))
                proj=rng.normal(size=(depth,1))*.2;bias=np.zeros(1)
                t=torch.tensor(feat,requires_grad=True)
                actual=decode(tuple(torch.tensor(states)),t,torch.tensor(previous),
                              [{k:torch.tensor(v) for k,v in w.items()} for w in cells],
                              torch.tensor(proj),torch.tensor(bias),None if attn is None else torch.tensor(attn))
                expected=reference(list(states),feat,previous,cells,proj,bias,attn)
                for a,b in zip(actual[:2],expected[:2]):np.testing.assert_allclose(a.detach(),b,atol=1e-13,rtol=1e-12)
                for a,b in zip(actual[2],expected[2]):np.testing.assert_allclose(a.detach(),b,atol=1e-13,rtol=1e-12)
                counts['rollouts']+=1
                actual[0].sum().backward()
                for index in [(0,0,0),(1,days-1,2)]:
                    plus=feat.copy();minus=feat.copy();plus[index]+=1e-5;minus[index]-=1e-5
                    a=reference(list(states),plus,previous,cells,proj,bias,attn)[0].sum()
                    b=reference(list(states),minus,previous,cells,proj,bias,attn)[0].sum()
                    np.testing.assert_allclose(t.grad[index],(a-b)/2e-5,atol=1e-10,rtol=1e-6)
                    counts['gradient_checks']+=1
        for enc in [1,2,3]:
            state=torch.tensor(rng.normal(size=(enc,2,4)))
            for dec in [1,2,3]:
                result=convert_encoder_state(state,dec)
                expected=list(state[-dec:]) if enc>=dec else list(state)+[torch.zeros_like(state[0])]*(dec-enc)
                for a,b in zip(result,expected):torch.testing.assert_close(a,b)
                assert len(result)==dec;counts['state_mappings']+=1
    paths=['sciona/webtraffic_decoder.py','scripts/validate_webtraffic_decoder.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                equations_source='https://raw.githubusercontent.com/tensorflow/tensorflow/v1.10.0/tensorflow/contrib/rnn/python/ops/gru_ops.py',
                limitations=['Float64 numerical recurrence oracle and autograd checks; no TensorFlow/cuDNN runtime parity.',
                             'Dropout, source initialization and checkpoint conversion remain unimplemented.',
                             's32 attention is disabled in decoder; optional attention input tested for source generality.',
                             'Default s32 depth267 and full encoder/training lifecycle remain to be integrated.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_decoder.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

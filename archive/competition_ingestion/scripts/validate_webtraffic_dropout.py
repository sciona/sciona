"""Synthetic deterministic dropout-mask recurrence checks against NumPy equations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sciona.webtraffic_dropout import apply_mask,decode_with_masks
from sciona.webtraffic_decoder import decode
from scripts.validate_webtraffic_decoder import numpy_step


def validate(root,source):
    source_pin=json.loads((root/'docs/reviews/competition_webtraffic_dropout_source.json').read_text())
    assert hashlib.sha256((source/'tf110_rnn_cell_impl.py').read_bytes()).hexdigest()==source_pin['sha256']
    counts=dict(masked_rollouts=0,no_dropout_parity=0,separate_state_output_checks=0,rejections=0)
    for seed in range(8):
        rng=np.random.RandomState(312+seed)
        for layers in [1,2]:
            days,batch,depth=9,2,4
            features=rng.normal(size=(batch,days,3));previous=rng.normal(size=batch)
            states=rng.normal(size=(layers,batch,depth));cells=[];masks=[];keeps=[]
            for layer in range(layers):
                ins=4
                cells.append(dict(gate_kernel=rng.normal(size=(ins+depth,depth*2))*.1,
                                  gate_bias=np.ones(depth*2),candidate_kernel=rng.normal(size=(ins+depth,depth))*.1,
                                  candidate_bias=np.zeros(depth)))
                keep=dict(input=.8,state=.6,output=.7);keeps.append(keep)
                masks.append({k:rng.uniform(size=(days,batch,ins if k=='input' else depth))<p for k,p in keep.items()})
            proj=rng.normal(size=(depth,1));bias=np.zeros(1)
            tcells=[{k:torch.tensor(v) for k,v in w.items()} for w in cells]
            tmasks=[{k:torch.tensor(v) for k,v in m.items()} for m in masks]
            actual=decode_with_masks(tuple(torch.tensor(states)),torch.tensor(features),torch.tensor(previous),
                                     tcells,torch.tensor(proj),torch.tensor(bias),tmasks,keeps)
            expected=[];raws=[];ss=list(states);prev=previous[:,None]
            for day in range(days):
                value=np.concatenate([prev,features[:,day]],axis=1);updated=[]
                for h,w,m,k in zip(ss,cells,masks,keeps):
                    raw=numpy_step(value/k['input']*m['input'][day],h,w)
                    updated.append(raw/k['state']*m['state'][day])
                    value=raw/k['output']*m['output'][day]
                ss=updated;prev=value.dot(proj)+bias;expected.append(prev[:,0]);raws.append(value)
            np.testing.assert_allclose(actual[0],np.stack(expected),atol=1e-12)
            np.testing.assert_allclose(actual[1],np.stack(raws),atol=1e-12)
            for a,b in zip(actual[2],ss):np.testing.assert_allclose(a,b,atol=1e-12)
            counts['masked_rollouts']+=1
            keep_one=[dict(input=1.,state=1.,output=1.) for _ in cells]
            a=decode_with_masks(tuple(torch.tensor(states)),torch.tensor(features),torch.tensor(previous),tcells,
                                torch.tensor(proj),torch.tensor(bias),[{} for _ in cells],keep_one)
            b=decode(tuple(torch.tensor(states)),torch.tensor(features),torch.tensor(previous),tcells,torch.tensor(proj),torch.tensor(bias))
            for av,bv in zip(a[:2],b[:2]):torch.testing.assert_close(av,bv,rtol=0,atol=0)
            counts['no_dropout_parity']+=1
        # Entire state dropped while output survives: projection must remain nonzero.
        mask=dict(input=None,state=torch.zeros(1,batch,depth,dtype=torch.bool),output=torch.ones(1,batch,depth,dtype=torch.bool))
        a=decode_with_masks((torch.tensor(states[0]),),torch.tensor(features[:,:1]),torch.tensor(previous),tcells[:1],
                            torch.tensor(proj),torch.tensor(bias),[mask],[dict(input=1.,state=.5,output=.5)])
        assert torch.count_nonzero(a[2][0])==0 and torch.count_nonzero(a[1])>0
        counts['separate_state_output_checks']+=1
    for mask,p in [(None,.5),(torch.ones(2),.5),(torch.ones(3,dtype=torch.bool),.5),(None,0),(None,1.1)]:
        try:apply_mask(torch.ones(2),mask,p)
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Invalid dropout contract accepted')
    x=torch.tensor([2.,4.],requires_grad=True)
    apply_mask(x,torch.tensor([True,False]),.5).sum().backward()
    torch.testing.assert_close(x.grad,torch.tensor([2.,0.]))
    paths=['sciona/webtraffic_dropout.py','scripts/validate_webtraffic_dropout.py','sciona/webtraffic_decoder.py',
           'scripts/validate_webtraffic_decoder.py','docs/reviews/competition_webtraffic_dropout_source.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Explicit boolean masks only; TensorFlow RNG/variational mask generation not reproduced.',
                             'Source s32 uses nonvariational decoder dropout; caller must supply one mask per time step.',
                             'Encoder-to-decoder gate dropout and full-width training integration remain.',
                             'No original TensorFlow runtime or checkpoint parity claim.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_dropout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))

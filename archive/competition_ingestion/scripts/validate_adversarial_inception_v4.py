"""Pinned source topology and full actual V4 main/aux differentiation."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_inception_v4 import build_inception_v4
from trace_adversarial_inception_v4 import trace


def main():
    torch.set_num_threads(2)
    topology=json.loads((ROOT/'sciona/adversarial_inception_v4_topology.json').read_text())
    assert topology==trace()
    nodes={n['name']:n for n in topology['nodes']}
    shapes={'Conv2d_1a_3x3':[149,149,32],'Mixed_3a':[73,73,160],
            'Mixed_4a':[71,71,192],'Mixed_5a':[35,35,384],
            **{'Mixed_5'+c:[35,35,384] for c in 'bcde'},
            **{'Mixed_6'+c:[17,17,1024] for c in 'abcdefgh'},
            **{'Mixed_7'+c:[8,8,1536] for c in 'abcd'},
            'AuxLogits':[1001],'Logits':[1001]}
    for name,shape in shapes.items(): assert nodes[topology['endpoints'][name]]['shape'][1:]==shape
    aux=nodes['InceptionV4/AuxLogits/AvgPool_1a_5x5']
    assert aux['inputs']==[topology['endpoints']['Mixed_6h']]
    assert aux['kernel']==[5,5] and aux['stride']==3 and aux['padding']=='VALID'
    aux_conv=nodes['InceptionV4/AuxLogits/Conv2d_2a']
    assert aux_conv['kernel']==[5,5] and aux_conv['out_channels']==768 and aux_conv['padding']=='VALID'
    assert sum(n['op']=='linear' for n in nodes.values())==2
    rng=torch.get_rng_state().clone()
    model=build_inception_v4(seed=811).requires_grad_(False)
    assert torch.equal(rng,torch.get_rng_state())
    before={k:v.clone() for k,v in model.named_buffers()}
    x=torch.linspace(-1,1,3*299*299).reshape(1,3,299,299).requires_grad_()
    logits,endpoints=model(x)
    assert logits.shape==endpoints['AuxLogits'].shape==(1,1001)
    assert torch.isfinite(logits).all() and torch.isfinite(endpoints['AuxLogits']).all()
    torch.testing.assert_close(endpoints['Predictions'].sum(1),torch.ones(1))
    a=F.cross_entropy(logits,torch.tensor([17]));b=F.cross_entropy(endpoints['AuxLogits'],torch.tensor([17]))
    ga,=torch.autograd.grad(a,x,retain_graph=True)
    gb,=torch.autograd.grad(b,x,retain_graph=True)
    gc,=torch.autograd.grad(a+.4*b,x)
    for g in (ga,gb,gc): assert torch.isfinite(g).all() and torch.count_nonzero(g)>0
    torch.testing.assert_close(gc,ga+.4*gb,rtol=2e-4,atol=1e-15)
    with torch.no_grad():
        y,e=model.train()(x.detach())
        assert torch.equal(y,logits.detach()) and torch.equal(e['AuxLogits'],endpoints['AuxLogits'].detach())
    assert all(torch.equal(v,before[k]) for k,v in model.named_buffers())
    other=build_inception_v4(seed=812,state=model.state_dict()).requires_grad_(False)
    with torch.no_grad():
        y,e=other(x.detach())
        assert torch.equal(y,logits.detach()) and torch.equal(e['AuxLogits'],endpoints['AuxLogits'].detach())
    assert torch.equal(rng,torch.get_rng_state())
    rejected=0
    for call in [lambda:build_inception_v4(seed=-1),lambda:build_inception_v4(seed=1,state={}),
                 lambda:model(torch.zeros(1,3,75,75))]:
        try:call()
        except ValueError:rejected+=1
        else:raise AssertionError('invalid model boundary accepted')
    paths=['sciona/adversarial_inception_v4.py','sciona/adversarial_inception_v4_topology.json',
           'sciona/adversarial_inception_ops.py','sciona/adversarial_normalization.py',
           'scripts/trace_adversarial_inception_v4.py','scripts/trace_adversarial_inception_v3.py',
           'scripts/validate_adversarial_inception_v4.py','docs/reviews/competition_adversarial_source_pins.json',
           'docs/reviews/competition_adversarial_inception_ops.json','docs/licenses/Adversarial-non_targeted-Apache-2.0.txt']
    report={'format':'adversarial-inception-v4-validation.v1','result':'passed',
            'checks':{'pinned_source_topology_retrace':True,'operations':len(nodes),
                      'convolutions':sum(n['op']=='conv' for n in nodes.values()),'independent_endpoint_shapes':len(shapes),
                      'full_299_main_and_aux_input_gradients':True,'weighted_gradient_linearity':True,
                      'frozen_buffers_modes_rng':True,'state_reload_neural_execution_exact':True,
                      'invalid_cases_rejected':rejected,'parameters':sum(p.numel() for p in model.parameters())},
            'limits':'Shape-only source AST tracing with explicit Slim/TF adapters; separately validated Torch CPU primitives. Synthetic single-example float32 full network. No historical TensorFlow binary/initializer or checkpoint mapping, pretrained quality or full ensemble attack claim. Max-pool tie-gradient parity unproven.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_inception_v4.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

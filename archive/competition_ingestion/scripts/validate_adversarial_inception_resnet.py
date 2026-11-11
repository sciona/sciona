"""Pinned residual topology and full synthetic Inception-ResNet-v2 execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_inception_resnet import build_inception_resnet
from trace_adversarial_inception_resnet import trace


def main():
    torch.set_num_threads(2)
    topology=json.loads((ROOT/'sciona/adversarial_inception_resnet_topology.json').read_text())
    assert topology==trace()
    other=trace('targeted')
    assert topology['nodes']==other['nodes'] and topology['endpoints']==other['endpoints']
    nodes={n['name']:n for n in topology['nodes']}
    shapes={'Conv2d_1a_3x3':[149,149,32],'Conv2d_2a_3x3':[147,147,32],
            'Conv2d_2b_3x3':[147,147,64],'MaxPool_3a_3x3':[73,73,64],
            'Conv2d_3b_1x1':[73,73,80],'Conv2d_4a_3x3':[71,71,192],
            'MaxPool_5a_3x3':[35,35,192],'Mixed_5b':[35,35,320],
            'Mixed_6a':[17,17,1088],'PreAuxLogits':[17,17,1088],
            'Mixed_7a':[8,8,2080],'Conv2d_7b_1x1':[8,8,1536],
            'AuxLogits':[1001],'Logits':[1001]}
    for name,shape in shapes.items():assert nodes[topology['endpoints'][name]]['shape'][1:]==shape
    scales=[n for n in nodes.values() if n['op']=='scale']
    assert Counter(n['factor'] for n in scales)=={.17:10,.10:20,.20:9,1.:1}
    assert sum(n['op']=='add' for n in nodes.values())==40
    assert sum(n['op']=='relu' for n in nodes.values())==39
    for n in scales:
        up=nodes[n['inputs'][0]]
        assert up['op']=='conv' and up['normalized'] is False and up['activation'] is False
    last=next(n for n in scales if n['factor']==1.)
    last_add=next(n for n in nodes.values() if n['op']=='add' and last['name'] in n['inputs'])
    assert nodes['InceptionResnetV2/Conv2d_7b_1x1']['inputs']==[last_add['name']]
    aux=nodes['InceptionResnetV2/AuxLogits/Conv2d_1a_3x3']
    assert aux['op']=='pool' and aux['kernel']==[5,5] and aux['stride']==3 and aux['padding']=='VALID'
    assert aux['inputs']==[topology['endpoints']['PreAuxLogits']]
    rng=torch.get_rng_state().clone()
    model=build_inception_resnet(seed=831).requires_grad_(False)
    assert torch.equal(rng,torch.get_rng_state())
    buffers={k:v.clone() for k,v in model.named_buffers()}
    x=torch.linspace(-1,1,3*299*299).reshape(1,3,299,299).requires_grad_()
    logits,ends=model(x)
    assert logits.shape==ends['AuxLogits'].shape==(1,1001)
    assert torch.isfinite(logits).all() and torch.isfinite(ends['AuxLogits']).all()
    torch.testing.assert_close(ends['Predictions'].sum(1),torch.ones(1))
    a=F.cross_entropy(logits,torch.tensor([17]));b=F.cross_entropy(ends['AuxLogits'],torch.tensor([17]))
    ga,=torch.autograd.grad(a,x,retain_graph=True)
    gb,=torch.autograd.grad(b,x,retain_graph=True)
    gc,=torch.autograd.grad(a+.4*b,x)
    for g in (ga,gb,gc):assert torch.isfinite(g).all() and torch.count_nonzero(g)>0
    torch.testing.assert_close(gc,ga+.4*gb,rtol=2e-4,atol=1e-12)
    with torch.no_grad():
        y,e=model.train()(x.detach())
        assert torch.equal(y,logits.detach()) and torch.equal(e['AuxLogits'],ends['AuxLogits'].detach())
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    restored=build_inception_resnet(seed=832,state=model.state_dict()).requires_grad_(False)
    with torch.no_grad():
        y,e=restored(x.detach())
        assert torch.equal(y,logits.detach()) and torch.equal(e['AuxLogits'],ends['AuxLogits'].detach())
    assert torch.equal(rng,torch.get_rng_state())
    paths=['sciona/adversarial_inception_resnet.py','sciona/adversarial_inception_resnet_topology.json',
           'sciona/adversarial_inception_ops.py','sciona/adversarial_normalization.py',
           'scripts/trace_adversarial_inception_resnet.py','scripts/trace_adversarial_inception_v4.py',
           'scripts/trace_adversarial_inception_v3.py','scripts/validate_adversarial_inception_resnet.py',
           'docs/reviews/competition_adversarial_source_pins.json',
           'docs/reviews/competition_adversarial_inception_ops.json','docs/licenses/Adversarial-non_targeted-Apache-2.0.txt']
    report={'format':'adversarial-inception-resnet-validation.v1','result':'passed',
            'checks':{'both_pinned_source_topologies_match':True,'operations':len(nodes),
                      'convolutions':sum(n['op']=='conv' for n in nodes.values()),
                      'independent_endpoint_shapes':len(shapes),'residual_scale_and_activation_contracts':40,
                      'full_299_main_aux_input_gradients':True,'weighted_gradient_linearity':True,
                      'state_reload_neural_execution_exact':True,'frozen_buffers_modes_rng':True,
                      'parameters':sum(p.numel() for p in model.parameters())},
            'limits':'Synthetic single-example CPUfloat32, shape-only source AST adapters and separately validated Torch primitives. No historical TensorFlow binary/initializer or checkpoint equivalence, pretrained quality or full ensemble attack claim. Max-pool tie-gradient parity unresolved.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_inception_resnet.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

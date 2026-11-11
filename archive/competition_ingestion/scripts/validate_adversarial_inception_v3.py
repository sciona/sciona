"""Source topology retrace and full synthetic main/aux input differentiation."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_inception_v3 import build_inception_v3
from trace_adversarial_inception_v3 import trace


def main():
    torch.set_num_threads(2)
    topology = json.loads((ROOT/'sciona/adversarial_inception_v3_topology.json').read_text())
    assert topology == trace()
    other = trace('targeted')
    assert topology['nodes']==other['nodes'] and topology['endpoints']==other['endpoints']
    nodes = {n['name']:n for n in topology['nodes']}
    endpoint_shapes = {
        'Conv2d_1a_3x3':[149,149,32], 'MaxPool_5a_3x3':[35,35,192],
        'Mixed_5b':[35,35,256], 'Mixed_5c':[35,35,288], 'Mixed_5d':[35,35,288],
        'Mixed_6a':[17,17,768], 'Mixed_6b':[17,17,768], 'Mixed_6c':[17,17,768],
        'Mixed_6d':[17,17,768], 'Mixed_6e':[17,17,768], 'Mixed_7a':[8,8,1280],
        'Mixed_7b':[8,8,2048], 'Mixed_7c':[8,8,2048], 'AuxLogits':[1001], 'Logits':[1001]}
    for name, shape in endpoint_shapes.items(): assert nodes[topology['endpoints'][name]]['shape'][1:]==shape
    aux_pool = nodes['InceptionV3/AuxLogits/AvgPool_1a_5x5']
    assert aux_pool['inputs']==[topology['endpoints']['Mixed_6e']]
    assert aux_pool['kernel']==[5,5] and aux_pool['stride']==3 and aux_pool['padding']=='VALID'
    for name in ['InceptionV3/AuxLogits/Conv2d_2b_1x1','InceptionV3/Logits/Conv2d_1c_1x1']:
        assert nodes[name]['normalized'] is False and nodes[name]['activation'] is False
    rng = torch.get_rng_state().clone()
    model = build_inception_v3(seed=731)
    assert torch.equal(rng,torch.get_rng_state())
    # Freeze model state only, preserving image differentiation.
    model.requires_grad_(False)
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    x = torch.linspace(-1,1,3*299*299).reshape(1,3,299,299).requires_grad_()
    logits, endpoints = model(x)
    assert logits.shape==endpoints['AuxLogits'].shape==(1,1001)
    assert torch.isfinite(logits).all() and torch.isfinite(endpoints['AuxLogits']).all()
    torch.testing.assert_close(endpoints['Predictions'].sum(dim=1),torch.ones(1))
    main_loss = F.cross_entropy(logits,torch.tensor([17]))
    aux_loss = F.cross_entropy(endpoints['AuxLogits'],torch.tensor([17]))
    g_main, = torch.autograd.grad(main_loss,x,retain_graph=True)
    g_aux, = torch.autograd.grad(aux_loss,x,retain_graph=True)
    g_combined, = torch.autograd.grad(main_loss+.4*aux_loss,x)
    for g in (g_main,g_aux,g_combined): assert torch.isfinite(g).all() and torch.count_nonzero(g)>0
    torch.testing.assert_close(g_combined,g_main+.4*g_aux,rtol=2e-4,atol=1e-15)
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    with torch.no_grad():
        repeated, repeated_ends = model.train()(x.detach())
        assert torch.equal(repeated,logits.detach())
        assert torch.equal(repeated_ends['AuxLogits'],endpoints['AuxLogits'].detach())
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    # Strict complete state reload followed by actual neural execution.
    restored = build_inception_v3(seed=732,state=model.state_dict()).requires_grad_(False)
    with torch.no_grad():
        y, e = restored(x.detach())
        assert torch.equal(y,logits.detach()) and torch.equal(e['AuxLogits'],endpoints['AuxLogits'].detach())
    assert torch.equal(rng,torch.get_rng_state())
    paths = ['sciona/adversarial_inception_v3.py','sciona/adversarial_inception_v3_topology.json',
             'sciona/adversarial_inception_ops.py','sciona/adversarial_normalization.py',
             'scripts/trace_adversarial_inception_v3.py','scripts/validate_adversarial_inception_v3.py',
             'docs/reviews/competition_adversarial_source_pins.json',
             'docs/reviews/competition_adversarial_inception_ops.json',
             'docs/licenses/Adversarial-non_targeted-Apache-2.0.txt']
    report = {'format':'adversarial-inception-v3-validation.v1','result':'passed',
              'checks':{'pinned_both_branch_topology_retrace':True,'operations':len(nodes),
                        'convolutions':sum(n['op']=='conv' for n in nodes.values()),
                        'independent_endpoint_shapes':len(endpoint_shapes),
                        'full_299_main_and_aux_input_gradients':True,'weighted_gradient_linearity':True,
                        'state_reload_neural_execution_exact':True,'frozen_buffers_modes_rng':True,
                        'parameters':sum(p.numel() for p in model.parameters())},
              'limits':'Topology traced from exact source functions with shape-only Slim/TF adapters. Runtime uses separately validated Torch primitives. No whole-network TensorFlow binary comparison, historical initializer or TF checkpoint mapping, pretrained quality or complete attack claim. Synthetic single-example CPU float32 execution. Max-pool tie-gradient parity unresolved.',
              'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_inception_v3.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()

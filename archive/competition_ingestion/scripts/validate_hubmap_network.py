"""Execute full candidate/source network parity on synthetic tensors and states."""
import argparse
import ast
from collections import OrderedDict
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace
import torch

from sciona.hubmap_network import UNET_SERESNEXT101


def source_model(root, source_root):
    pins = json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    encoder = 'src/pretrained-models.pytorch-master/pretrainedmodels/models/senet.py'
    for name in [encoder, 'src/models.py']:
        assert hashlib.sha256((source_root/name).read_bytes()).hexdigest() == pins['files'][name]
    ns = dict(nn=torch.nn, OrderedDict=OrderedDict, math=math)
    tree = ast.parse((source_root/encoder).read_text())
    names = {'SEModule','Bottleneck','SEResNeXtBottleneck','SENet','se_resnext101_32x4d'}
    nodes = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<pinned-encoder>', 'exec'), ns)
    network_ns = dict(torch=torch,nn=torch.nn,F=torch.nn.functional,
        pretrainedmodels=SimpleNamespace(se_resnext101_32x4d=ns['se_resnext101_32x4d']))
    nodes = [n for n in ast.parse((source_root/'src/models.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<pinned-network>', 'exec'), network_ns)
    return network_ns['UNET_SERESNEXT101']


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [tensor for item in value for tensor in tensors(item)]


def exact_state(a, b):
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)


def validate(root, source_root, training=False, heads=True, side=32):
    torch.set_num_threads(1)
    original = source_model(root, source_root)
    torch.manual_seed(109)
    candidate = UNET_SERESNEXT101((side,side),heads,heads,None,load_weights=False)
    candidate_rng = torch.get_rng_state()
    torch.manual_seed(109)
    reference = original((side,side),heads,heads,None,load_weights=False)
    torch.testing.assert_close(candidate_rng, torch.get_rng_state(), rtol=0, atol=0)
    exact_state(candidate.state_dict(), reference.state_dict())
    candidate.train(training); reference.train(training)
    torch.manual_seed(113)
    image = torch.randn(2,3,side,side)
    a = image.clone().requires_grad_(training)
    b = image.clone().requires_grad_(training)
    with torch.set_grad_enabled(training):
        result = tensors(candidate(a)); expected = tensors(reference(b))
        assert len(result) == len(expected) == (6 if heads else 1)
        for x,y in zip(result,expected):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        if training:
            # Include all active heads in an independently formed scalar loss.
            sum((i+1)*x.square().mean() for i,x in enumerate(result)).backward()
            sum((i+1)*x.square().mean() for i,x in enumerate(expected)).backward()
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
            for (name,p),(other,q) in zip(candidate.named_parameters(),reference.named_parameters()):
                assert name == other and (p.grad is None) == (q.grad is None)
                if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)
    exact_state(candidate.state_dict(),reference.state_dict())
    count = sum(p.numel() for p in candidate.parameters())
    paths = ['sciona/hubmap_encoder.py','sciona/hubmap_network.py','scripts/validate_hubmap_network.py',
             'docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,training=training,deep_supervision=heads,classification_head=heads,
        spatial_size=side,batch=2,parameters=count,exact_initial_state_and_rng=True,exact_outputs=True,
        exact_input_and_parameter_gradients=training,exact_final_buffers=True,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Synthetic initialized weights and float32 CPU tensors; no learned accuracy or historical numerical environment claim.',
                     'Optional source threshold shortcut rejected; original training configurations disable it.',
                     'Complete optimizer/augmentation/tiling/pseudo-label/inference graph remains.'])


if __name__ == '__main__':
    signal.alarm(120)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--training',action='store_true')
    parser.add_argument('--no-heads',action='store_true')
    parser.add_argument('--side',type=int,default=32)
    args = parser.parse_args();root=Path(__file__).resolve().parents[1]
    report = validate(root,args.source_root,args.training,not args.no_heads,args.side)
    name = 'competition_hubmap_network_' + ('train' if args.training else 'eval') + ('_plain' if args.no_heads else '_heads') + '_' + str(args.side) + '.json'
    (root/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}},indent=2))

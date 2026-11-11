"""Full-sized synthetic forward/backward comparison with pinned image networks."""
import argparse
import ast
import functools
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from sciona.bengali_gan_networks import BengaliGenerator, BengaliDiscriminator, initialize_image_network


def main(source,output):
    raw=source.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest != '600d64d01226bd5ca879a347ca06f148193fd73c88039222cee1832c847bc716':
        raise ValueError('reference hash mismatch')
    text=raw.decode()
    parsed=ast.parse(text[text.index('class ResnetGenerator('):text.index('class ImagePool():')])
    names={'ResnetGenerator','ResnetBlock','NLayerDiscriminator','init_weight'}
    nodes=[n for n in parsed.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('missing reference definitions')
    namespace=dict(torch=torch,nn=nn,init=nn.init,functools=functools)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-image-network-reference>','exec'),namespace)
    torch.set_num_threads(2)
    norm=functools.partial(nn.InstanceNorm2d,affine=False,track_running_stats=False)
    results=[]
    for kind in ['generator','discriminator']:
        actual=BengaliGenerator() if kind=='generator' else BengaliDiscriminator()
        reference=(namespace['ResnetGenerator'](3,3,64,norm,False,9) if kind=='generator'
                   else namespace['NLayerDiscriminator'](3,64,3,norm))
        initialize_image_network(actual,generator=torch.Generator().manual_seed(831))
        torch.manual_seed(831)
        namespace['init_weight'](reference,.02)
        left,right=list(actual.parameters()),list(reference.parameters())
        if len(left)!=len(right):
            raise AssertionError('parameter inventory differs')
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        inputs=torch.randn(1,3,224,224,generator=torch.Generator().manual_seed(44))
        x,y=inputs.clone().requires_grad_(),inputs.clone().requires_grad_()
        a,b=actual(x),reference(y)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        if not torch.isfinite(x.grad).all() or not x.grad.abs().sum()>0:
            raise AssertionError('invalid input gradient')
        for ap,bp in zip(left,right):
            torch.testing.assert_close(ap.grad,bp.grad,rtol=0,atol=0)
            if not torch.isfinite(ap.grad).all():
                raise AssertionError('nonfinite parameter gradient')
        results.append(dict(network=kind,parameters=sum(p.numel() for p in left),output_shape=list(a.shape),
            exact_initialization=True,exact_forward=True,exact_input_and_parameter_gradients=True))
        del actual,reference,left,right,a,b,x,y
    report=dict(passed=True,synthetic_only=True,catalog_mutations=0,networks=results,source_only_sha256=digest,
        sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                ['sciona/bengali_gan_networks.py','scripts/validate_bengali_network_reference.py']},
        limits=['Full source-sized image architectures only; EfficientNet classifier and complete training remain pending.',
                'Both sides use the installed PyTorch runtime; this is not historical runtime qualification.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.source,args.output)

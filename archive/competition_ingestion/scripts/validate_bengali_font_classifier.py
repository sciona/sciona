"""Source-head equivalence and actual B0 gradient probe, without pretrained weights."""
import argparse
import ast
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from efficientnet_pytorch import EfficientNet
import torch
from torch import nn

from sciona.bengali_font_classifier import BengaliFontClassifier
from sciona.bengali_font_guidance import FrozenFontGuidance


def main(source,output):
    raw=source.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest != '600d64d01226bd5ca879a347ca06f148193fd73c88039222cee1832c847bc716':
        raise ValueError('reference hash mismatch')
    text=raw.decode()
    parsed=ast.parse(text[text.index('class BengalModel('):text.index('norm_layer = functools.partial')])
    nodes=[n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='BengalModel']
    if len(nodes)!=1:
        raise ValueError('missing source classifier')
    namespace=dict(torch=torch,nn=nn)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-font-head-reference>','exec'),namespace)
    torch.set_num_threads(2)
    torch.manual_seed(624)
    backbone=EfficientNet.from_name('efficientnet-b0')
    second=copy.deepcopy(backbone)
    torch.manual_seed(953)
    actual=BengaliFontClassifier(backbone).eval()
    torch.manual_seed(953)
    reference=namespace['BengalModel'](second,hidden_size=1280,class_num=14784).eval()
    x=torch.rand(1,3,224,224,generator=torch.Generator().manual_seed(56)).requires_grad_()
    y=x.detach().clone().requires_grad_()
    actual.requires_grad_(False)
    reference.requires_grad_(False)
    logits,expected=actual(x),reference(y)
    torch.testing.assert_close(logits,expected,rtol=0,atol=0)
    nn.functional.cross_entropy(logits,torch.tensor([14783])).backward()
    nn.functional.cross_entropy(expected,torch.tensor([14783])).backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    del reference,second,expected,y
    before={name:value.clone() for name,value in actual.state_dict().items()}
    guidance=FrozenFontGuidance(actual,weight=1.)
    guidance.train()
    image=x.detach().clone().requires_grad_()
    loss=guidance(image,torch.tensor([14783]))
    loss.backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum()>0
    assert all(p.grad is None for p in actual.parameters())
    for name,value in actual.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert all(not m.training for m in actual.modules())
    report=dict(passed=True,synthetic_only=True,catalog_mutations=0,
        backbone='efficientnet-b0',installed_efficientnet_pytorch=version('efficientnet-pytorch'),
        pretrained_weights=False,output_shape=list(logits.shape),exact_source_head_logits=True,
        exact_source_head_input_gradients=True,frozen_parameters_and_buffers_unchanged=True,
        finite_nonzero_guidance_gradient=True,source_only_sha256=digest,
        sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                ['sciona/bengali_font_classifier.py','scripts/validate_bengali_font_classifier.py']},
        limits=['Head equivalence uses the same installed B0 implementation on both sides; original notebook dependency identity remains unqualified.',
                'Random weights and synthetic images verify computation and gradients, not font recognition or trained checkpoint fidelity.',
                'Full pipeline training and promotion remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.source,args.output)

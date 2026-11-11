"""Full installed B7 with source-described heads; no pretrained/training claim."""
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

from sciona.grapheme_b7_head import GraphemeB7Head
from sciona.bengali_ood_loss import ood_training_loss


def main(source, output):
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != '600d64d01226bd5ca879a347ca06f148193fd73c88039222cee1832c847bc716':
        raise ValueError('source head reference content differs')
    text = raw.decode()
    tree = ast.parse(text[text.index('class BengalModel('):text.index('norm_layer = functools.partial')])
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BengalModel']
    if len(nodes) != 1:
        raise ValueError('source head class absent')
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-parameterized-head>', 'exec'), namespace)
    torch.set_num_threads(2)
    results = []
    for branch, classes in [('ood', 1295), ('seen', 14784)]:
        torch.manual_seed(905)
        backbone = EfficientNet.from_name('efficientnet-b7')
        other = copy.deepcopy(backbone)
        torch.manual_seed(906)
        actual = GraphemeB7Head(backbone, branch=branch).eval()
        torch.manual_seed(906)
        reference = namespace['BengalModel'](other, hidden_size=2560, class_num=classes).eval()
        # Differentiate through the full feature extractor to the input, while
        # retaining parameter gradients only for the two classifier head layers.
        backbone.requires_grad_(False)
        other.requires_grad_(False)
        x = torch.rand(1, 3, 137, 236, generator=torch.Generator().manual_seed(907)).requires_grad_()
        y = x.detach().clone().requires_grad_()
        logits, expected = actual(x), reference(y)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        labels = torch.tensor([classes - 1])
        objective = ood_training_loss if branch == 'ood' else nn.functional.cross_entropy
        loss, expected_loss = objective(logits, labels), objective(expected, labels)
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        for layer in ['ln', 'fc']:
            for name in ['weight', 'bias']:
                torch.testing.assert_close(getattr(getattr(actual, layer), name).grad,
                                           getattr(getattr(reference, layer), name).grad, rtol=0, atol=0)
        result = dict(branch=branch, classes=classes, output_shape=list(logits.shape),
            backbone_parameters=sum(p.numel() for p in backbone.parameters()),
            logits_exact=True, input_gradients_exact=True, head_parameter_gradients_exact=True,
            finite_nonzero_input_gradient=True)
        results.append(result)
        print(json.dumps(result), flush=True)
        del actual, reference, backbone, other, x, y, logits, expected, loss, expected_loss
    files = ['sciona/grapheme_b7_head.py', 'scripts/validate_bengali_b7_heads.py', 'sciona/bengali_ood_loss.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        pretrained_weights=False, full_installed_efficientnet_b7=True, branches=results,
        versions={name: version(name) for name in ['torch', 'efficientnet-pytorch']},
        source_head_sha256=source_sha,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Parameterized public B0 head configured with winner-described B7 feature/output widths; not recovered B7 training code.',
                'Both sides share installed B7 implementation; winner dependency version and ImageNet checkpoint are not qualified.',
                'Synthetic tensors with random model initialization; normalization, warmup duration, full training and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'approved': False, 'catalog_mutations': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)

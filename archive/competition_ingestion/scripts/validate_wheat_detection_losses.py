"""Compare historical proposal/ROI objectives and gradients with pinned source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler
from torchvision.models.detection.roi_heads import fastrcnn_loss as installed_roi_loss

from sciona.wheat_detection_losses import rpn_loss, roi_loss


def main(root, output):
    manifest = json.loads((root / 'manifest.json').read_text())
    sources = {}
    definitions = {}
    for file in ('rpn.py', 'roi_heads.py'):
        name = 'models/detection/' + file
        raw = (root / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != next(row['sha256'] for row in manifest if row['file'] == name):
            raise ValueError('historical loss source drift')
        sources[name] = digest
        tree = ast.parse(raw)
        if file == 'rpn.py':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RegionProposalNetwork')
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_loss')
        else:
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fastrcnn_loss')
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical-loss>', 'exec'), namespace)
        definitions[file] = namespace[node.name]
    torch.set_num_threads(2)
    sampler = BalancedPositiveNegativeSampler(256, .5)
    cases, modern_roi_difference = 0, False
    for seed in range(32):
        generator = torch.Generator().manual_seed(940 + seed)
        counts = [311, 277, 69]
        for kind in ('rpn', 'roi'):
            labels = [torch.randint(-1 if kind == 'rpn' else 0, 2, (n,), generator=generator) for n in counts]
            if seed == 0:
                labels = [torch.zeros_like(value) for value in labels]
            if kind == 'rpn':
                labels = [value.float() for value in labels]
            targets = [torch.randn(n, 4, generator=generator) * .2 for n in counts]
            logits = torch.randn((sum(counts),) if kind == 'rpn' else (sum(counts), 2), generator=generator).requires_grad_()
            deltas = (torch.randn(sum(counts), 4 if kind == 'rpn' else 8, generator=generator) * .2).requires_grad_()
            ref_logits, ref_deltas = logits.detach().clone().requires_grad_(), deltas.detach().clone().requires_grad_()
            torch.manual_seed(seed)
            if kind == 'rpn':
                observed = rpn_loss(logits, deltas, labels, targets, sampler)
            else:
                observed = roi_loss(logits, deltas, labels, targets)
                modern_roi_difference |= not torch.equal(observed[1], installed_roi_loss(logits, deltas, labels, targets)[1])
            torch.manual_seed(seed)
            expected = (definitions['rpn.py'](SimpleNamespace(fg_bg_sampler=sampler), ref_logits, ref_deltas, labels, targets)
                        if kind == 'rpn' else definitions['roi_heads.py'](ref_logits, ref_deltas, labels, targets))
            for a, b in zip(observed, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                assert torch.isfinite(a)
            sum(observed).backward()
            sum(expected).backward()
            for a, b in ((logits, ref_logits), (deltas, ref_deltas)):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                assert torch.isfinite(a.grad).all()
            cases += 1
    assert modern_roi_difference
    files = ['sciona/wheat_detection_losses.py', 'scripts/validate_wheat_detection_losses.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=sources, exact_loss_and_gradient_cases=cases, all_background_cases_included=True,
        installed_roi_default_difference_observed=True,
        regression_contract=dict(rpn='L1', roi='Smooth L1 beta=1.0'),
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Loss boundary only; source sampler equivalence and full detector integration remain separate.',
                'Both paths use installed Torch kernels; no historical native-kernel equivalence claim.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.output)

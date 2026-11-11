"""Verify publisher checkpoint initialization and both complete B7 head paths."""
import argparse
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path

import torch
from torch import nn

from sciona.grapheme_b7_initialization import load_standard_b7, STANDARD_B7_SHA256, STANDARD_B7_URL
from sciona.grapheme_b7_head import GraphemeB7Head
from sciona.bengali_ood_loss import ood_training_loss


def main(checkpoint, output):
    torch.set_num_threads(2)
    payload = checkpoint.read_bytes()
    if hashlib.sha256(payload).hexdigest() != STANDARD_B7_SHA256:
        raise ValueError('reference content differs')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    results = []
    for branch, classes in [('ood', 1295), ('seen', 14784)]:
        torch.manual_seed(982)
        backbone = load_standard_b7(checkpoint)
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        assert all(p.requires_grad for p in backbone.parameters()) and backbone.training
        classifier = GraphemeB7Head(backbone, branch=branch).eval()
        x = torch.rand(1, 3, 137, 236, generator=torch.Generator().manual_seed(983)).requires_grad_()
        logits = classifier(x)
        assert logits.shape == (1, classes) and torch.isfinite(logits).all()
        labels = torch.tensor([classes - 1])
        objective = ood_training_loss if branch == 'ood' else nn.functional.cross_entropy
        loss = objective(logits, labels)
        loss.backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        # Original ImageNet classification layer is intentionally unused by
        # extract_features; it must not receive gradients from the new task head.
        unused = {'backbone._fc.weight', 'backbone._fc.bias'}
        for name, parameter in classifier.named_parameters():
            if name in unused:
                assert parameter.grad is None
            else:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert backbone._conv_stem.weight.grad.abs().sum() > 0
        assert classifier.fc.weight.grad.abs().sum() > 0
        # Evaluation/backpropagation alone must leave checkpoint tensors intact.
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        results.append(dict(branch=branch, classes=classes, checkpoint_tensors_exact=len(state),
            finite_loss=True, finite_input_and_parameter_gradients=True,
            nonzero_stem_and_task_head_gradients=True, original_imagenet_head_unused=True,
            checkpoint_state_unchanged=True))
        print(json.dumps(results[-1]), flush=True)
        del classifier, backbone, logits, loss, x
    files = ['sciona/grapheme_b7_initialization.py', 'sciona/grapheme_b7_head.py',
             'sciona/bengali_ood_loss.py', 'scripts/validate_bengali_b7_initialization.py']
    report = dict(passed=True, synthetic_only=True, approved=False, catalog_mutations=0,
        checkpoint_url=STANDARD_B7_URL, checkpoint_sha256=STANDARD_B7_SHA256,
        checkpoint_bytes=len(payload), tensor_only_loading=True, branches=results,
        versions={name: version(name) for name in ['torch', 'efficientnet-pytorch']},
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Publisher standard ImageNet checkpoint candidate; exact winner artifact identity is unproven.',
                'Random task heads and synthetic unnormalized tensors; no recognition accuracy or training completion claim.',
                'Original image normalization, warmup duration, dependency identity, artifact licensing and full publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'approved': False, 'catalog_mutations': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.output)

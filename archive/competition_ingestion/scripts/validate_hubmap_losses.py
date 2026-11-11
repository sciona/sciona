"""Compare synthetic loss values/gradients against pinned original statements."""
import argparse
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import signal
import numpy as np
import torch
from sciona.hubmap_losses import training_loss


def reference(source_root, pins):
    for name in ['src/lovasz_loss.py', 'src/losses.py', 'src/02_train/run.py']:
        assert hashlib.sha256((source_root / name).read_bytes()).hexdigest() == pins['files'][name]
    ns = dict(torch=torch, F=torch.nn.functional, Variable=torch.autograd.Variable,
              np=np, ifilterfalse=itertools.filterfalse)
    for name, functions in [('src/lovasz_loss.py', {'lovasz_grad', 'lovasz_hinge', 'lovasz_hinge_flat', 'flatten_binary_scores', 'mean'}),
                            ('src/losses.py', {'criterion_lovasz_hinge_non_empty'})]:
        tree = ast.parse((source_root / name).read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
        assert {n.name for n in selected} == functions
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-source-loss>', 'exec'), ns)
    tree = ast.parse((source_root / 'src/02_train/run.py').read_text())
    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for i, statement in enumerate(node.body):
                if isinstance(statement, ast.Assign) and ast.unparse(statement) == 'loss = criterion(logits, y_true)':
                    blocks.append(node.body[i:])
    assert len(blocks) == 1 and len(blocks[0]) == 4
    wrapper = ast.parse('''def source_loss(logits, y_true, logits_deeps, logits_clf, y_clf, config):
    batch, c, h, w = y_true.shape
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_clf = torch.nn.BCEWithLogitsLoss()
''').body[0]
    wrapper.body.extend(copy.deepcopy(blocks[0]))
    wrapper.body.append(ast.Return(value=ast.Name(id='loss', ctx=ast.Load())))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), '<pinned-training-statements>', 'exec'), ns)
    return ns['source_loss']


def validate(root, source_root):
    torch.set_num_threads(1)
    pins = json.loads((root / 'docs/reviews/competition_hubmap_source_pins.json').read_text())
    source = reference(source_root, pins)
    reports = []
    for batch, height, width in [(1, 1, 1), (2, 3, 4), (3, 5, 7)]:
        for foreground in ['empty', 'full', 'mixed']:
            for deep_count, classify in [(0, False), (2, False), (0, True), (4, True)]:
                torch.manual_seed(73)
                target = (torch.rand(batch, 1, height, width) > .6).float()
                if foreground == 'empty': target.zero_()
                if foreground == 'full': target.fill_(1)
                if foreground == 'mixed' and batch > 1: target[0].zero_(); target[-1].fill_(1)
                tensors = [torch.randn_like(target) for _ in range(deep_count + 1)]
                if classify: tensors.append(torch.randn(batch, 1))
                labels = (torch.arange(batch) % 2).float()
                candidate = [v.clone().requires_grad_() for v in tensors]
                original = [v.clone().requires_grad_() for v in tensors]
                actual = training_loss(candidate[0], target, deep_logits=candidate[1:1+deep_count],
                    classification_logits=candidate[-1] if classify else None, classification_targets=labels if classify else None)
                expected = source(original[0], target, original[1:1+deep_count], original[-1] if classify else None,
                    labels, dict(deepsupervision=bool(deep_count), clfhead=classify))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.backward(); expected.backward()
                for a, b in zip(candidate, original):
                    assert (a.grad is None) == (b.grad is None)
                    if a.grad is not None: torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                reports.append(dict(batch=batch, geometry=[height,width], foreground=foreground,
                    deep_heads=deep_count, classification_head=classify, exact_value_and_gradients=True))
    paths = ['sciona/hubmap_losses.py', 'scripts/validate_hubmap_losses.py', 'tests/test_hubmap_losses.py',
             'docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False, synthetic_only=True, source_commit=pins['commit'], cases=reports,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Float32 CPU loss and all input-logit gradients only; no mixed precision or historical-library claim.',
                     'Model forward, optimizer loop, preparation, pseudo-labeling and inference remain separate execution obligations.',
                     'Invalid tensor contracts reject explicitly; no predictive-quality claim.'])


if __name__ == '__main__':
    signal.alarm(90)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate(root, args.source_root)
    (root / 'docs/reviews/competition_hubmap_losses.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cases=len(report['cases']), exact_values_and_gradients=True, approved=False)))

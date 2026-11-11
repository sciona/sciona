"""Synthetic detector loss and gradient parity against the pinned source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_losses import detector_loss, LearnedNoisyOR, classifier_loss


def validate(root, source_root):
    pins = json.loads((root / 'docs/reviews/competition_dsb_source_pins.json').read_text())
    source = checked_source(source_root, pins, 'layers.py')
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ['hard_mining', 'Loss']]
    class ScalarDiagnostics(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if (isinstance(node.value, ast.Attribute) and node.value.attr == 'data'
                    and isinstance(node.slice, ast.Constant) and node.slice.value == 0):
                return ast.copy_location(ast.Call(func=ast.Attribute(value=node.value.value,
                    attr='item', ctx=ast.Load()), args=[], keywords=[]), node)
            return node
    tree = ast.fix_missing_locations(ScalarDiagnostics().visit(tree))
    scope = dict(torch=torch, nn=torch.nn)
    exec(compile(tree, '<pinned-loss-modern-scalar-diagnostics>', 'exec'), scope)
    generator = torch.Generator().manual_seed(42)
    cases = 0
    max_loss_error = max_gradient_error = 0.
    for batch in [1, 3]:
        for positives in [False, True]:
            for training in [False, True]:
                for hard in [0, 2, 50]:
                    raw = torch.randn((batch, 9, 5), generator=generator, dtype=torch.float64)
                    labels = torch.randn(raw.shape, generator=generator, dtype=torch.float64) * .1
                    labels[..., 0] = -1
                    labels[:, 1, 0] = 0
                    if positives: labels[:, 0, 0] = 1
                    source_input, candidate_input = raw.clone().requires_grad_(), raw.clone().requires_grad_()
                    expected = scope['Loss'](hard)(source_input, labels, train=training)
                    actual = detector_loss(candidate_input, labels, num_hard=hard, training=training)
                    expected[0].backward()
                    actual['total'].backward()
                    torch.testing.assert_close(actual['total'], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(actual['classification'], raw.new_tensor(expected[1]), rtol=0, atol=0)
                    torch.testing.assert_close(actual['regression'], raw.new_tensor(expected[2:6]), rtol=0, atol=0)
                    torch.testing.assert_close(candidate_input.grad, source_input.grad, rtol=0, atol=0)
                    assert [actual[k] for k in ['positive_correct', 'positive_count', 'negative_correct', 'negative_count']] == [int(v) for v in expected[6:]]
                    max_loss_error = max(max_loss_error, abs(float(actual['total'].detach() - expected[0].detach())))
                    max_gradient_error = max(max_gradient_error, float((candidate_input.grad-source_input.grad).abs().max()))
                    cases += 1
    network = ast.parse(checked_source(source_root, pins, 'net_classifier.py'))
    case_class = next(n for n in network.body if isinstance(n, ast.ClassDef) and n.name == 'CaseNet')
    forward = next(n for n in case_class.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    assignments = [n for n in forward.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ['base_prob', 'casePred']]
    assert len(assignments) == 2
    aggregation = compile(ast.Module(body=assignments, type_ignores=[]), '<pinned-noisy-or>', 'exec')
    training_source = checked_source(source_root, pins, 'training/classifier/trainval_classifier.py')
    training_tree = ast.parse(training_source)
    # Select the training loss assignments directly, retaining original reductions.
    function = next(n for n in training_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_casenet')
    names = ['loss2', 'missMask', 'missLoss', 'loss']
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    assignments.sort(key=lambda n: n.lineno)
    assert [n.targets[0].id for n in assignments] == names
    classifier_reference = compile(ast.Module(body=assignments, type_ignores=[]), '<pinned-classifier-loss>', 'exec')
    classifier_cases = 0
    for dtype in [torch.float32, torch.float64]:
        for batch, proposals in [(1, 1), (2, 5)]:
            for baseline in [-30., 0., 3.]:
                probabilities = torch.linspace(.005, .1, batch * proposals, dtype=dtype).reshape(batch, proposals)
                known = torch.ones_like(probabilities)
                known.reshape(-1)[-1] = 0
                truth = torch.arange(batch, dtype=dtype) % 2
                original = probabilities.clone().requires_grad_()
                original_baseline = torch.tensor([baseline], dtype=dtype, requires_grad=True)
                source_scope = dict(torch=torch, out=original, self=SimpleNamespace(baseline=original_baseline))
                exec(aggregation, source_scope)
                source_scope.update(binary_cross_entropy=torch.nn.functional.binary_cross_entropy,
                    y=truth[:, None], casePred_each=original, isnod=known, xsize=original.shape,
                    args=SimpleNamespace(miss_thresh=.03, miss_ratio=1.))
                exec(classifier_reference, source_scope)
                candidate = probabilities.clone().requires_grad_()
                model = LearnedNoisyOR().to(dtype=dtype)
                with torch.no_grad(): model.baseline.fill_(baseline)
                actual = classifier_loss(model(candidate), candidate, truth, known)
                for key, reference_name in [('total','loss'), ('classification','loss2'), ('miss','missLoss')]:
                    torch.testing.assert_close(actual[key], source_scope[reference_name], rtol=0, atol=0)
                source_scope['loss'].backward()
                actual['total'].backward()
                torch.testing.assert_close(candidate.grad, original.grad, rtol=0, atol=0)
                torch.testing.assert_close(model.baseline.grad, original_baseline.grad, rtol=0, atol=0)
                classifier_cases += 1
    paths = ['sciona/dsb_losses.py', 'tests/test_dsb_losses.py', 'scripts/validate_dsb_losses.py']
    return dict(approved=False, synthetic_only=True, source_commit=pins['commit'],
                detector_loss_gradient_parity_cases=cases, max_loss_error=max_loss_error,
                max_gradient_error=max_gradient_error,
                classifier_loss_gradient_parity_cases=classifier_cases,
                classifier_exact_loss_probability_and_baseline_gradient_parity=True,
                limitations=['Source diagnostic tensor.data[0] becomes tensor.item() for current scalar tensors.',
                             'Parity uses the same installed PyTorch; historical dependency fidelity is not proven.',
                             'No trained model, full graph, or catalog approval is implied.',
                             'Empty negative sets are explicitly rejected instead of returning the source undefined mean.'],
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, args.source_root)
    (root/'docs/reviews/competition_dsb_loss_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

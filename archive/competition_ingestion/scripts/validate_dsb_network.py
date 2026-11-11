"""Pinned architecture/state/forward/backward checks with synthetic weights."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path

import torch

from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_network import DetectorNet, CaseNet


def reference(root, pins, file, class_names):
    layers = ast.parse(checked_source(root, pins, 'layers.py'))
    nodes = [n for n in layers.body if isinstance(n, ast.ClassDef) and n.name == 'PostRes']
    network = ast.parse(checked_source(root, pins, file))
    nodes += [n for n in network.body if isinstance(n, ast.ClassDef) and n.name in class_names]
    tree = ast.Module(body=nodes, type_ignores=[])
    for n in ast.walk(tree):
        if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and isinstance(n.left, ast.Subscript)
                and isinstance(n.left.value, ast.Name) and n.left.value.id == 'featshape'):
            n.op = ast.FloorDiv()
    scope = dict(torch=torch, nn=torch.nn, config={'anchors': [10, 30, 60]})
    exec(compile(tree, '<pinned-network-integer-slices>', 'exec'), scope)
    return scope


def validate(root, source_root):
    torch.set_num_threads(1)
    pins = json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    sources = [reference(source_root, pins, 'net_detector.py', {'Net'}),
               reference(source_root, pins, 'net_classifier.py', {'Net', 'CaseNet'})]
    cases = []
    for classifier in [False, True]:
        for training in [False, True]:
            for side in [16, 32]:
                torch.manual_seed(101)
                actual = CaseNet(2) if classifier else DetectorNet()
                original = sources[1]['CaseNet'](2) if classifier else sources[0]['Net']()
                buffer = io.BytesIO()
                torch.save(actual.state_dict(), buffer)
                buffer.seek(0)
                restored = torch.load(buffer, weights_only=True)
                original.load_state_dict(restored, strict=True)
                actual.load_state_dict(restored, strict=True)
                actual.train(training); original.train(training)
                shape = (1, 2, 1, side, side, side) if classifier else (2, 1, side, side, side)
                coord_shape = (1, 2, 3, side//4, side//4, side//4) if classifier else (2, 3, side//4, side//4, side//4)
                x = torch.randn(shape); coord = torch.randn(coord_shape)
                ax, ac = x.clone().requires_grad_(), coord.clone().requires_grad_()
                rx, rc = x.clone().requires_grad_(), coord.clone().requires_grad_()
                torch.manual_seed(991)
                expected = original(rx, rc)
                torch.manual_seed(991)
                observed = actual(ax, ac)
                if not classifier: expected, observed = (expected,), (observed,)
                for a, b in zip(observed, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                sum(t.square().mean() for t in expected).backward()
                sum(t.square().mean() for t in observed).backward()
                torch.testing.assert_close(ax.grad, rx.grad, rtol=0, atol=0)
                torch.testing.assert_close(ac.grad, rc.grad, rtol=0, atol=0)
                for (name, a), (other_name, b) in zip(actual.named_parameters(), original.named_parameters()):
                    assert name == other_name
                    assert (a.grad is None) == (b.grad is None)
                    if a.grad is not None: torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
                cases.append(dict(classifier=classifier, training=training, side=side,
                                  strict_state_roundtrip=True, exact_output_input_and_parameter_gradient_parity=True))
                del actual, original, expected, observed
    # Source classifier's actual 96-cube inference geometry, synthetic weights.
    torch.manual_seed(11)
    actual, original = CaseNet(1).eval(), sources[1]['CaseNet'](1).eval()
    original.load_state_dict(actual.state_dict(), strict=True)
    with torch.inference_mode():
        x, coord = torch.zeros(1, 1, 1, 96, 96, 96), torch.zeros(1, 1, 3, 24, 24, 24)
        expected, observed = original(x, coord), actual(x, coord)
        for a, b in zip(observed, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
    paths = ['sciona/dsb_network.py', 'tests/test_dsb_network.py', 'scripts/validate_dsb_network.py']
    return dict(approved=False, synthetic_weights_and_inputs=True, source_commit=pins['commit'],
                network_parity_cases=cases, source_classifier_96_cube_inference_parity=True,
                limitations=['No trained checkpoint or predictive accuracy validation.',
                             'Same current PyTorch for source and candidate, not historical runtime fidelity.',
                             '96-cube parity covers inference; gradient cases use 16/32 cubes.',
                             'Graph assembly, coordinate generation, proposal decoding and full pipeline execution remain.'],
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, args.source_root)
    (root/'docs/reviews/competition_dsb_network_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

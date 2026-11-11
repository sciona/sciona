"""Compare normalization outputs and input gradients with pinned source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from sciona.wheat_frozen_normalization import source_frozen_batch_norm


SOURCE_SHA256 = 'f75578efe5c4054e2fdd479db15bfb5cca15b610b909185b40f452c4b87a91db'


def main(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('historical normalization source drift')
    node = next(n for n in ast.parse(raw).body
                if isinstance(n, ast.ClassDef) and n.name == 'FrozenBatchNorm2d')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-normalization>', 'exec'), namespace)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(731)
    cases = 0
    default_difference_observed = False
    for dtype in (torch.float32, torch.float64):
        for channels in (64, 256, 512, 1024, 2048):
            for low_variance in (False, True):
                reference = namespace['FrozenBatchNorm2d'](channels).to(dtype)
                with torch.no_grad():
                    reference.weight.copy_(torch.randn(channels, generator=generator, dtype=dtype))
                    reference.bias.copy_(torch.randn(channels, generator=generator, dtype=dtype))
                    reference.running_mean.copy_(torch.randn(channels, generator=generator, dtype=dtype))
                    variance = torch.rand(channels, generator=generator, dtype=dtype) + .01
                    reference.running_var.copy_(variance * (1e-6 if low_variance else 1.0))
                actual = source_frozen_batch_norm(channels).to(dtype)
                actual.load_state_dict(reference.state_dict(), strict=True)
                x = torch.randn(2, channels, 3, 5, generator=generator, dtype=dtype).requires_grad_()
                y = x.detach().clone().requires_grad_()
                expected, observed = reference(x), actual(y)
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                upstream = torch.randn(expected.shape, generator=generator, dtype=dtype)
                expected.backward(upstream)
                observed.backward(upstream)
                torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
                assert torch.isfinite(observed).all() and torch.isfinite(y.grad).all()
                assert not list(actual.parameters())
                default = type(actual)(channels).to(dtype)
                default.load_state_dict(reference.state_dict(), strict=True)
                default_difference_observed |= not torch.equal(default(x.detach()), expected.detach())
                cases += 1
    assert default_difference_observed
    files = ['sciona/wheat_frozen_normalization.py', 'scripts/validate_wheat_frozen_normalization.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=SOURCE_SHA256, source_url='https://raw.githubusercontent.com/pytorch/vision/v0.5.0/torchvision/ops/misc.py',
        exact_output_and_input_gradient_cases=cases, installed_default_difference_observed=True,
        required_eps=0.0, installed_torch_version=torch.__version__,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Normalization component only; no full Faster R-CNN initialization or training qualification.',
                'Synthetic positive variances; no competition data or accuracy claim.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)

"""Compare complete pyramid initialization, forward and backward with source."""
import argparse
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from sciona.wheat_feature_pyramid import WheatFeaturePyramid


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'd5acedbbc2b192516044c9d655e1757dfecea662c518b6de9dfc5056f50f8d2e':
        raise ValueError('historical feature pyramid source drift')
    names = {'FeaturePyramidNetwork', 'ExtraFPNBlock', 'LastLevelMaxPool'}
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name in names]
    namespace = dict(torch=torch, nn=nn, Tensor=Tensor, F=F, OrderedDict=OrderedDict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-pyramid-source>', 'exec'), namespace)
    torch.set_num_threads(2)
    results = []
    for seed, sizes in [(813, [(200, 200), (100, 100), (50, 50), (25, 25)]),
                        (814, [(19, 23), (10, 12), (5, 6), (3, 3)])]:
        torch.manual_seed(seed)
        actual = WheatFeaturePyramid()
        torch.manual_seed(seed)
        reference = namespace['FeaturePyramidNetwork']([256, 512, 1024, 2048], 256,
            extra_blocks=namespace['LastLevelMaxPool']())
        assert set(actual.state_dict()) == set(reference.state_dict())
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
        generator = torch.Generator().manual_seed(seed + 100)
        x = OrderedDict((str(i), torch.randn(1, c, *s, generator=generator).requires_grad_())
                        for i, (c, s) in enumerate(zip((256, 512, 1024, 2048), sizes)))
        y = OrderedDict((k, v.detach().clone().requires_grad_()) for k, v in x.items())
        observed, expected = actual(x), reference(y)
        assert list(observed) == list(expected) == ['0', '1', '2', '3', 'pool']
        for key in observed:
            torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0)
        sum(v.square().mean() for v in observed.values()).backward()
        sum(v.square().mean() for v in expected.values()).backward()
        for key in x:
            torch.testing.assert_close(x[key].grad, y[key].grad, rtol=0, atol=0)
            assert torch.isfinite(x[key].grad).all()
        for (key, p), (reference_key, q) in zip(actual.named_parameters(), reference.named_parameters()):
            assert key == reference_key
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            assert torch.isfinite(p.grad).all()
        results.append(dict(input_spatial_shapes=sizes, initialized_state_exact=True,
            output_shapes={k: list(v.shape) for k, v in observed.items()},
            outputs_and_all_input_parameter_gradients_exact=True))
    files = ['sciona/wheat_feature_pyramid.py', 'scripts/validate_wheat_feature_pyramid.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=digest, cases=results,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Feature pyramid only; complete detector losses, proposal selection and ROI heads remain pending.',
                'Historical source uses installed Torch kernels; no historical native-kernel equivalence claim.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)

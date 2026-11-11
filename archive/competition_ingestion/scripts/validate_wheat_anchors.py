"""Verify complete ordered anchor arrays against historical torchvision source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from sciona.wheat_anchors import WheatAnchorGenerator


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != '62ea420ac8bba44d91d3b27f46c627fcc6b91c15ae706142fcbe079c10a31751':
        raise ValueError('historical anchor source drift')
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'AnchorGenerator')
    namespace = dict(torch=torch, torchvision=torchvision, nn=torch.nn, Optional=Optional, List=List, Dict=Dict)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical-anchors>', 'exec'), namespace)
    torch.set_num_threads(2)
    args = (((32,), (64,), (128,), (256,), (512,)), ((.5, 1., 2.),) * 5)
    cases, default_difference = [], False
    for dtype in (torch.float16, torch.float32, torch.float64):
        for height, width in ((800, 800), (832, 1344)):
            features = [torch.empty(2, 1, (height + s - 1) // s, (width + s - 1) // s, dtype=dtype)
                        for s in (4, 8, 16, 32, 64)]
            images = ImageList(torch.empty(2, 3, height, width), [(height, width), (height - 17, width - 23)])
            actual, reference = WheatAnchorGenerator(), namespace['AnchorGenerator'](*args)
            for repeat in range(2):
                observed, expected = actual(images, features), reference(images, features)
                assert len(observed) == len(expected) == 2
                for a, b in zip(observed, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0, check_dtype=True)
                    assert torch.isfinite(a).all()
            modern = AnchorGenerator(*args)(images, features)
            default_difference |= modern[0].dtype != expected[0].dtype or not torch.equal(modern[0], expected[0])
            cases.append(dict(feature_dtype=str(dtype), canvas=[height, width], batch_size=2,
                anchors_per_image=len(observed[0]), output_dtype=str(observed[0].dtype), repeated_calls_exact=True))
    assert default_difference
    files = ['sciona/wheat_anchors.py', 'scripts/validate_wheat_anchors.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=digest, cases=cases, installed_default_difference_observed=True,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Anchor arrays only; proposal matching, sampling and filtering remain separate.',
                'CPU eager execution; no CUDA, tracing or historical native-kernel equivalence claim.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)

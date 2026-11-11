"""Compare detector transforms with historical resize dispatch and source code."""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Dict, Optional
import warnings

import torch
import torchvision
from torchvision.models.detection.image_list import ImageList

from sciona.wheat_image_transform import WheatImageTransform


def main(root, output):
    files = {
        'models/detection/transform.py': '46024143aa8d85210135e659fc6487b642cd5d8b0296cc4b7f29f1a329a836aa',
        'torch14_functional.py': '45d25e49078f0c00bd559d3d7ff412d9250d6f4ad1faf406bb4abab968db4cfb'}
    raw = {}
    for name, digest in files.items():
        raw[name] = (root / name).read_bytes()
        if hashlib.sha256(raw[name]).hexdigest() != digest:
            raise ValueError('historical transform source drift')
    interpolate = next(n for n in ast.parse(raw['torch14_functional.py']).body
                       if isinstance(n, ast.FunctionDef) and n.name == 'interpolate')
    old_functional = dict(torch=torch, math=math, warnings=warnings, __package__='torch.nn')
    exec(compile(ast.Module(body=[interpolate], type_ignores=[]), '<torch14-interpolate>', 'exec'), old_functional)
    class TorchProxy:
        nn = SimpleNamespace(functional=SimpleNamespace(interpolate=old_functional['interpolate']))
        def __getattr__(self, name):
            return getattr(torch, name)
    nodes = [n for n in ast.parse(raw['models/detection/transform.py']).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {'GeneralizedRCNNTransform', 'resize_boxes'}]
    namespace = dict(torch=TorchProxy(), torchvision=torchvision, nn=torch.nn, math=math,
                     ImageList=ImageList, List=List, Tuple=Tuple, Dict=Dict, Optional=Optional)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-transform>', 'exec'), namespace)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(1191)
    cases = []
    shapes = [(1024, 1024), (375, 1001), (701, 431)]
    for training in (True, False):
        for dtype in (torch.float32, torch.float64):
            actual = WheatImageTransform().train(training)
            reference = namespace['GeneralizedRCNNTransform'](800, 1333, [.485, .456, .406], [.229, .224, .225]).train(training)
            images = [torch.rand(3, h, w, generator=generator, dtype=dtype) for h, w in shapes]
            ref_images = [image.detach().clone() for image in images]
            targets = [dict(boxes=torch.tensor([[0., 0., w-1., h-1.], [13., 17., 77., 99.]], dtype=dtype),
                            labels=torch.ones(2, dtype=torch.int64)) for h, w in shapes]
            ref_targets = copy.deepcopy(targets)
            torch.manual_seed(1192)
            observed, observed_targets = actual(images, targets)
            rng = torch.get_rng_state()
            torch.manual_seed(1192)
            expected, expected_targets = reference(ref_images, ref_targets)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            assert observed.image_sizes == expected.image_sizes
            assert observed_targets is targets and expected_targets is ref_targets
            torch.testing.assert_close(observed.tensors, expected.tensors, rtol=0, atol=0)
            for a, b in zip(targets, ref_targets):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            # Historical batch iterator cannot be differentiated on modern Torch.
            # Qualify normalization/resize gradients separately from exact padding values.
            for image in images:
                x = image.detach().clone().requires_grad_()
                y = image.detach().clone().requires_grad_()
                torch.manual_seed(1193)
                a, _ = actual.resize(actual.normalize(x), None)
                torch.manual_seed(1193)
                b, _ = reference.resize(reference.normalize(y), None)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.square().mean().backward()
                b.square().mean().backward()
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                assert torch.isfinite(x.grad).all()
            predictions, ref_predictions = copy.deepcopy(targets), copy.deepcopy(ref_targets)
            a = actual.postprocess(predictions, observed.image_sizes, shapes)
            b = reference.postprocess(ref_predictions, expected.image_sizes, shapes)
            for x, y in zip(a, b):
                torch.testing.assert_close(x['boxes'], y['boxes'], rtol=0, atol=0)
            cases.append(dict(training=training, dtype=str(dtype), input_shapes=shapes,
                resized_shapes=observed.image_sizes, padded_shape=list(observed.tensors.shape),
                images_targets_postprocessing_and_rng_exact=True, normalization_resize_gradients_exact=True))
    implementation = ['sciona/wheat_image_transform.py', 'scripts/validate_wheat_image_transform.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=files, cases=cases,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in implementation},
        limits=['Bounding-box detector only; masks and keypoints are outside this competition implementation.',
                'Historical interpolate dispatch calls installed native kernels with explicit output size; old native-kernel parity is not claimed.',
                'Historical batch-padding iterator rejects autograd on modern Torch; padding values and normalization/resize gradients are qualified separately.',
                'Full detector execution and ROI pooling remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.output)

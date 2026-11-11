"""Exact source TTA replay on synthetic tensors and box coordinates."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sciona.wheat_tta import WheatTTA, VIEWS


def main(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '0a0c4a66d960b07aa5e56fe51c78b9a4081ae5f977e4969435596562c155e3bd':
        raise ValueError('TTA source differs')
    names = ['BaseWheatTTA', 'TTAHorizontalFlip', 'TTAVerticalFlip', 'TTARotate90', 'TTACompose']
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name in names]
    if len(nodes) != 5:
        raise ValueError('source TTA classes absent')
    namespace = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-wheat-tta>', 'exec'), namespace)
    torch.set_num_threads(2)
    cases = 0
    for size in [17, 512, 640, 768, 1024]:
        images = torch.arange(3 * size * size, dtype=torch.float32).reshape(1, 3, size, size)
        original = images.clone()
        for view, flags in enumerate(VIEWS):
            transforms = [namespace[name](size) for name, enabled in zip(names[1:4], flags) if enabled]
            reference = namespace['TTACompose'](transforms)
            actual = WheatTTA(view=view, image_size=size)
            torch.testing.assert_close(actual.augment_tensor(images), reference.effdet_augment(images), rtol=0, atol=0)
            torch.testing.assert_close(actual.augment_fasterrcnn([images[0]])[0], reference.fasterrcnn_augment([images[0]])[0], rtol=0, atol=0)
            for dtype in [np.float32, np.float64]:
                # Includes reversed endpoints and coordinates beyond image edges;
                # source ordering happens here and clipping belongs downstream.
                boxes = np.array([[1., 2., size - 3., size - 4.],
                                  [size, size, 0., 0.], [-1., 3., size + 2., 7.]], dtype=dtype)
                before = boxes.copy()
                np.testing.assert_array_equal(actual.deaugment_boxes(boxes), reference.deaugment_boxes(boxes.copy()))
                np.testing.assert_array_equal(boxes, before)
                assert actual.deaugment_boxes(np.empty((0, 4), dtype=dtype)).shape == (0, 4)
                cases += 1
            torch.testing.assert_close(images, original, rtol=0, atol=0)
    files = ['sciona/wheat_tta.py', 'scripts/validate_wheat_tta.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        exact_box_cases=cases, exact_tensor_and_list_view_cases=40, all_eight_source_views=True,
        all_four_detector_resolutions=True, caller_inputs_unchanged=True,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Image permutations and inverse boxes only; detector inference and full pipeline remain pending.',
                'Current NumPy/Torch execute both paths; historical dependency equivalence is separate.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'implementation_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)

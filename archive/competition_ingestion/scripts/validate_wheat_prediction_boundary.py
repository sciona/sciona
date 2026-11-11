"""Replay the pinned detector postprocessing loop bodies on synthetic outputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sciona.wheat_prediction_boundary import prepare_prediction
from sciona.wheat_tta import VIEWS


def main(source_root, output):
    raw = (source_root / 'predict.py').read_bytes()
    tta_raw = (source_root / 'dataset.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'db2780716d39d8c9c0900c4f4fe309ad5b0243095038631dc6e51f8e8ff4265f':
        raise ValueError('prediction reference differs')
    if hashlib.sha256(tta_raw).hexdigest() != '0a0c4a66d960b07aa5e56fe51c78b9a4081ae5f977e4969435596562c155e3bd':
        raise ValueError('TTA reference differs')
    names = ['BaseWheatTTA', 'TTAHorizontalFlip', 'TTAVerticalFlip', 'TTARotate90', 'TTACompose']
    nodes = [n for n in ast.parse(tta_raw).body if isinstance(n, ast.ClassDef) and n.name in names]
    ns = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-tta>', 'exec'), ns)
    loops = {}
    for n in ast.walk(ast.parse(raw)):
        if isinstance(n, ast.For) and ast.unparse(n.target) in ('(image_id, o)', '(det, image_id)'):
            key = 'fasterrcnn' if ast.unparse(n.target) == '(image_id, o)' else 'effdet'
            loops[key] = compile(ast.Module(body=n.body[2:], type_ignores=[]), '<pinned-prediction-boundary>', 'exec')
    if set(loops) != {'effdet', 'fasterrcnn'}:
        raise ValueError('detector loop bodies absent')
    cases = 0
    for size in [512, 640, 768, 1024]:
        for view, flags in enumerate(VIEWS):
            transform = ns['TTACompose']([ns[name](size) for name, active in zip(names[1:4], flags) if active])
            for detector, code in loops.items():
                for dtype in [np.float32, np.float64]:
                    for empty in [False, True]:
                        boxes = np.array([[1., 2., 8., 9.], [-1., 3., size+2., 7.],
                                          [3., 4., 7., 8.], [0., 0., size, size]], dtype=dtype)
                        scores = np.array([.2, .25, .2501, .99], dtype=dtype)
                        if empty:
                            boxes, scores = boxes[:0], scores[:0]
                        identity = 'synthetic_query'
                        env = dict(np=np, boxes=boxes.copy(), scores=scores.copy(),
                            tta_transform=transform, args=SimpleNamespace(img_size=size), image_id=identity,
                            box_pred={identity: []}, score_pred={identity: []}, label_pred={identity: []})
                        exec(code, env)
                        actual = prepare_prediction(boxes, scores, detector=detector, view=view, image_size=size)
                        for value, name in zip(actual, ['boxes', 'scores', 'labels']):
                            np.testing.assert_array_equal(value, env[name])
                        cases += 1
    files = ['sciona/wheat_prediction_boundary.py', 'sciona/wheat_tta.py', 'scripts/validate_wheat_prediction_boundary.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        exact_source_boundary_cases=cases, both_detectors=True, all_source_resolutions_and_views=True,
        source_prediction_sha256=hashlib.sha256(raw).hexdigest(),
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Synthetic detector outputs only; learned prediction quality and detector execution are not qualified.',
                'Full training, inference integration, dependency qualification and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'implementation_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.output)

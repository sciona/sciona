"""Pinned source crop comparisons; detector stand-ins do not prove MTCNN execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.dfdc_faces import extract_faces


class Detector:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.images = []

    def detect(self, image, *, landmarks):
        assert landmarks is False
        self.images.append(np.asarray(image).copy())
        return next(self.outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((ROOT / 'docs/reviews/competition_dfdc_source_pins.json').read_text())
    source = args.source_root / 'kernel_utils.py'
    record = next(r for r in pins['files'] if r['path'] == 'kernel_utils.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256']
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'FaceExtractor']
    assert len(nodes) == 1
    constructor_calls = []
    namespace = {'np': np, 'Image': Image, 'os': os,
                 'MTCNN': lambda **kwargs: constructor_calls.append(kwargs)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-face-extractor>', 'exec'), namespace)
    cls = namespace['FaceExtractor']
    cls(lambda _: None)
    assert constructor_calls == [{'margin': 0, 'thresholds': [.7, .8, .8], 'device': 'cuda'}]
    cases = [
        (None, None), ([], []), ([[4.2, 3.4, 10.2, 9.4]], [.01]),
        ([None, [1., 1., 6., 7.]], [.9, .8]),
        ([[-1., -1., 20., 20.]], [.6]), ([[4., 4., 4., 4.]], [.7]),
        ([[1., 1., 6., 7.], [4., 3., 10., 9.]], [.2, .99]),
        ([[1., 1., 6., 7.], [4., 3., 10., 9.]], [.2]),
    ]
    count = 0
    for shape in ((24, 30, 3), (25, 31, 3)):
        frames = np.random.default_rng(817).integers(0, 256, (len(cases), *shape), dtype=np.uint8)
        original_detector, adapted_detector = Detector(cases), Detector(cases)
        original = cls.__new__(cls)
        original.video_read_fn = lambda _: (frames, list(range(len(frames))))
        original.detector = original_detector
        expected = original.process_videos('', [''], [0])
        actual = extract_faces(frames, adapted_detector)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert a['frame_index'] == b['frame_idx']
            assert a['scores'] == b['scores']
            assert len(a['faces']) == len(b['faces'])
            for face_a, face_b in zip(a['faces'], b['faces']):
                np.testing.assert_array_equal(face_a, face_b)
        for a, b in zip(original_detector.images, adapted_detector.images):
            np.testing.assert_array_equal(a, b)
        count += len(cases)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_dfdc_faces.py'], cwd=ROOT, check=True)
    files = ['sciona/dfdc_faces.py', 'tests/test_dfdc_faces.py', 'scripts/validate_dfdc_faces.py',
             'docs/reviews/competition_dfdc_source_pins.json']
    report = {
        'format': 'dfdc-face-preparation-validation.v1', 'result': 'passed', 'source_commit': pins['commit'],
        'checks': {'source_frame_cases': count, 'source_detector_image_comparisons': count,
                   'source_constructor_configuration': 1, 'independent_tests': 12},
        'retained': ['Half-size PIL resize without overriding source interpolation default.',
                     'Double coordinates then int truncation; one-third margins; original NumPy crop slicing.',
                     'No landmark alignment, additional score filter, or box reordering.',
                     'None boxes, empty results, degenerate crops and zip truncation characterized.'],
        'adaptations': ['Strict uint8 RGB/spatial and finite box contract.',
                        'Omit source unused bookkeeping dimensions overwritten by last face box.',
                        'Caller supplies detector explicitly; no construction, weights or downloads.'],
        'limits': 'Synthetic arrays and detector outputs only. No MTCNN network or full classifier execution; not a promotion gate.',
        'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
    }
    (ROOT / 'docs/reviews/competition_dfdc_faces.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()

"""Validate pinned software using generated arrays only; no model downloads."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona import dfdc_inference_primitives as implementation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    pins_path = ROOT / 'docs/reviews/competition_dfdc_source_pins.json'
    pins = json.loads(pins_path.read_text())
    assert pins['commit'] == '89c6290490bac96b29193a4061b3db9dd3933e36'
    for record in pins['files']:
        data = (args.source_root / record['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == record['git_blob_sha1']
    assert (ROOT / 'docs/licenses/DFDC-MIT.txt').read_bytes() == (args.source_root / 'LICENSE').read_bytes()
    names = {'confident_strategy', 'put_to_center', 'isotropically_resize_image'}
    original = ast.parse((args.source_root / 'kernel_utils.py').read_text())
    selected = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    current = ast.parse(Path(implementation.__file__).read_text())
    actual = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
    assert len(selected) == 3
    for n in selected:
        assert ast.dump(n) == ast.dump(actual[n.name])
    namespace = {'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-dfdc-functions>', 'exec'), namespace)
    rng = np.random.default_rng(704)
    confidence_cases = 0
    for dtype in (np.float16, np.float32, np.float64):
        for size in (1, 10, 20, 32, 127):
            for population in (rng.uniform(0, 1, size), rng.uniform(0, .19, size), rng.uniform(.81, 1, size)):
                values = population.astype(dtype)
                expected = namespace['confident_strategy'](values)
                got = implementation.confident_strategy(values)
                assert got.dtype == expected.dtype
                np.testing.assert_array_equal(got, expected)
                confidence_cases += 1
    spatial_cases = 0
    for shape in ((17, 29, 3), (29, 17, 3), (19, 19, 3), (5, 7, 3)):
        values = rng.integers(0, 256, shape, dtype=np.uint8)
        for size in (8, 19, 38):
            for name in ('put_to_center', 'isotropically_resize_image'):
                expected = namespace[name](values.copy(), size)
                got = getattr(implementation, name)(values.copy(), size)
                np.testing.assert_array_equal(got, expected)
                spatial_cases += 1
    tests = 'tests/test_dfdc_inference_primitives.py'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', tests], cwd=ROOT, check=True)
    paths = ['sciona/dfdc_inference_primitives.py', tests,
             'scripts/validate_dfdc_inference_primitives.py', 'docs/licenses/DFDC-MIT.txt',
             'docs/reviews/competition_dfdc_source_pins.json']
    report = {
        'format': 'dfdc-inference-primitives-validation.v1',
        'source_commit': pins['commit'], 'result': 'passed',
        'checks': {'source_ast_equalities': 3, 'confidence_source_comparisons': confidence_cases,
                   'spatial_source_comparisons': spatial_cases, 'independent_tests_passed': 16},
        'scope': 'Synthetic primitive checks only. No detector, classifier, training, graph execution or promotion claim.',
        'source_behavior_retained': [
            'Strict confidence thresholds and count/fraction gates; NumPy input precision retained.',
            'Resize long edge then truncate short dimension; area downsampling, cubic upsampling.',
            'Top-left crop before centered uint8 zero padding; odd remainder to bottom/right.',
        ],
        'limitations': [
            'Original primitive functions assume valid inputs; runtime boundary validation is outstanding.',
            'No original observations, frames, weights or fold assignments accessed.',
        ],
        'code_and_evidence_sha256': {p: sha(ROOT / p) for p in paths},
    }
    (ROOT / 'docs/reviews/competition_dfdc_inference_primitives.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()

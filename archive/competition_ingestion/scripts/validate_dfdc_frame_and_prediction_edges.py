"""Characterize pinned DFDC reader and inference control flow on synthetic inputs."""

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.dfdc_frame_sampling import select_frames


class HistoricalNumpy:
    # Local compatibility alias for the removed historical np.int; no global patch.
    int = int

    def __getattr__(self, name):
        return getattr(np, name)


class Capture:
    def __init__(self, rgb, *, grab_failure=None, retrieve_failure=None):
        self.rgb = rgb
        self.cursor = -1
        self.released = False
        self.grab_failure = grab_failure
        self.retrieve_failure = retrieve_failure

    def get(self, key):
        assert key == cv2.CAP_PROP_FRAME_COUNT
        return len(self.rgb)

    def grab(self):
        self.cursor += 1
        return self.cursor < len(self.rgb) and self.cursor != self.grab_failure

    def retrieve(self):
        if self.cursor == self.retrieve_failure:
            return False, None
        return True, self.rgb[self.cursor, :, :, ::-1].copy()

    def release(self):
        self.released = True


class Model:
    def __init__(self, probability):
        self.probability = probability
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x.clone())
        return torch.logit(torch.full((len(x), 1), self.probability, dtype=x.dtype))


class Extractor:
    def __init__(self, count):
        self.count = count

    def process_video(self, _):
        if self.count == 0:
            return []
        return [{'faces': [np.full((5, 7, 3), i % 256, dtype=np.uint8)
                           for i in range(self.count)]}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    pins_path = ROOT / 'docs/reviews/competition_dfdc_source_pins.json'
    pins = json.loads(pins_path.read_text())
    assert pins['commit'] == '89c6290490bac96b29193a4061b3db9dd3933e36'
    source_path = args.source_root / 'kernel_utils.py'
    record = next(r for r in pins['files'] if r['path'] == 'kernel_utils.py')
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == record['sha256']
    tree = ast.parse(source_path.read_text())
    names = {'VideoReader', 'predict_on_video', 'confident_strategy',
             'isotropically_resize_image', 'put_to_center'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == 5
    active_capture = None
    cv_proxy = SimpleNamespace(
        VideoCapture=lambda _: active_capture, CAP_PROP_FRAME_COUNT=cv2.CAP_PROP_FRAME_COUNT,
        COLOR_BGR2RGB=cv2.COLOR_BGR2RGB, cvtColor=cv2.cvtColor, resize=cv2.resize,
        INTER_AREA=cv2.INTER_AREA, INTER_CUBIC=cv2.INTER_CUBIC,
    )
    torch_proxy = SimpleNamespace(
        tensor=lambda value, device: torch.tensor(value, device='cpu'),
        sigmoid=torch.sigmoid, no_grad=torch.no_grad,
    )
    namespace = {'np': HistoricalNumpy(), 'cv2': cv_proxy, 'torch': torch_proxy,
                 'normalize_transform': Normalize([.485, .456, .406], [.229, .224, .225])}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-dfdc-edge-oracle>', 'exec'), namespace)
    reader = namespace['VideoReader'](verbose=False)
    comparisons = 0
    for count in (0, 1, 2, 10, 31, 32, 33, 64, 100):
        rgb = np.random.default_rng(502 + count).integers(0, 256, (count, 4, 6, 3), dtype=np.uint8)
        for requested in (1, 2, 32):
            active_capture = Capture(rgb)
            expected = reader.read_frames('', requested)
            actual = select_frames(rgb, requested)
            if expected is None:
                assert actual is None
                assert not active_capture.released  # Source early return leaks the capture.
            else:
                np.testing.assert_array_equal(actual[0], expected[0])
                assert actual[1] == expected[1]
                assert active_capture.released
            comparisons += 1
    rgb = np.zeros((5, 4, 6, 3), dtype=np.uint8)
    partials = 0
    for mode, point, expected_indices in (
            ('grab_failure', 0, None), ('retrieve_failure', 0, None),
            ('grab_failure', 2, [0, 1]), ('retrieve_failure', 2, [0, 1])):
        active_capture = Capture(rgb, **{mode: point})
        result = reader.read_frames('', 5)
        assert (None if result is None else result[1]) == expected_indices
        assert active_capture.released
        partials += 1
    predict = namespace['predict_on_video']
    strategy = namespace['confident_strategy']
    prediction_checks = 0
    for count, expected_score, expected_seen in ((0, .5, 0), (1, .5, 1), (2, .75, 2),
                                                 (127, .75, 127), (128, .75, 127), (140, .75, 127)):
        model = Model(.75)
        diagnostics = io.StringIO()
        with redirect_stdout(diagnostics):
            score = predict(Extractor(count), '', 32, 8, [model], strategy)
        assert abs(float(score) - expected_score) < .001
        assert (len(model.inputs[0]) if model.inputs else 0) == expected_seen
        assert bool(diagnostics.getvalue()) == (count == 1)
        if model.inputs:
            assert model.inputs[0].dtype == torch.float16
            # First synthetic face is black, including its centered padding.
            expected = torch.tensor([-.485 / .229, -.456 / .224, -.406 / .225]).half()
            torch.testing.assert_close(model.inputs[0][0, :, 0, 0], expected, rtol=0, atol=0)
        prediction_checks += 1
    score = predict(Extractor(2), '', 32, 8, [Model(.75), Model(.25)], strategy)
    assert abs(float(score) - .5) < .001
    prediction_checks += 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        score = predict(Extractor(2), '', 32, 8, [], strategy)
    assert np.isnan(score)
    prediction_checks += 1
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_dfdc_frame_sampling.py'], cwd=ROOT, check=True)
    paths = ['sciona/dfdc_frame_sampling.py', 'tests/test_dfdc_frame_sampling.py',
             'scripts/validate_dfdc_frame_and_prediction_edges.py',
             'docs/reviews/competition_dfdc_source_pins.json']
    report = {
        'format': 'dfdc-frame-prediction-edge-validation.v1', 'result': 'passed',
        'source_commit': pins['commit'],
        'checks': {'array_sampling_source_comparisons': comparisons, 'source_partial_decode_cases': partials,
                   'source_prediction_control_flow_cases': prediction_checks, 'independent_sampling_tests': 15},
        'observations': [
            'Uniform requests exceeding clip length retain only the first frame due to repeated-index cursor behavior.',
            'Empty source clips return None without releasing capture; array adaptation owns no capture.',
            'Partial decoder failures return the successfully decoded prefix or None; not covered by complete-array API.',
            'One face reaches the classifier but squeeze/index failure returns 0.5. No faces also return 0.5.',
            'Default inference accepts at most 127 faces; normalization and half casting precede model call.',
            'Empty model ensemble produces NaN in source and must be rejected at the future runtime boundary.',
        ],
        'adaptations': [
            'Array API receives caller-decoded RGB; source BGR-to-RGB path checked via synthetic capture.',
            'Local historical np.int alias in source harness; implementation uses built-in int.',
            'Source prediction tensor allocation redirected from CUDA to CPU for control-flow checks only.',
        ],
        'limits': 'No real media or weights. Synthetic capture and stand-in classifiers; not codec, detector, full-model, training, graph or promotion evidence.',
        'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    }
    (ROOT / 'docs/reviews/competition_dfdc_frame_prediction_edges.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()

"""Compare both image adapters with pinned source functions on synthetic planes."""
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from sciona.osic_image_preprocessing import inference_plane, training_plane
from scripts.review_osic_original_scope import ROOT, TRAIN, INFERENCE, REVISION, review

FUNCTION_PINS = {
    'training': '603e73e60d97715bf81a1e43c5b77274f48d76c5545c402c73cfa9a4535cb287',
    'inference': '4ff759ede08f197111d0aada45ef7800a0960e6982fb76143b42f377fccdf621',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory):
    scope = review(directory)
    cv2.setNumThreads(0)
    if cv2.getNumThreads() != 1:
        raise ValueError('OpenCV serial execution required')
    paths = [ROOT/'sciona/osic_image_preprocessing.py', Path(__file__),
             ROOT/'tests/test_osic_image_preprocessing.py', ROOT/'scripts/review_osic_original_scope.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    source_functions = {}
    decoded = SimpleNamespace()
    for mode, filename, cell in [('training', TRAIN, 17), ('inference', INFERENCE, 9)]:
        source = ''.join(json.loads((Path(directory)/filename).read_text())['cells'][cell]['source'])
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        if hashlib.sha256(ast.unparse(function).encode()).hexdigest() != FUNCTION_PINS[mode]:
            raise ValueError('Pinned image function changed')
        namespace = {'cv2': cv2, 'pydicom': SimpleNamespace(dcmread=lambda _: decoded)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-image-function>', 'exec'), namespace)
        source_functions[mode] = namespace['get_img']
    rng = np.random.default_rng(783)
    cases = []
    for dtype, shape in [(np.int16, (13, 19)), (np.uint16, (21, 11)),
                         (np.int16, (512, 512)), (np.uint16, (1, 1))]:
        low = -1200 if dtype == np.int16 else 0
        pixels = rng.integers(low, 3000, size=shape, dtype=dtype)
        original = pixels.copy()
        decoded.pixel_array = pixels
        decoded.RescaleIntercept = -137.5
        decoded.RescaleSlope = 1.75
        for mode in ('training', 'inference'):
            reference = source_functions[mode](None)
            actual = (training_plane(pixels, decoded.RescaleIntercept, decoded.RescaleSlope)
                      if mode == 'training' else inference_plane(pixels))
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal(pixels, original)
            if actual.shape != (512, 512) or actual.dtype != np.float64 or np.shares_memory(actual, pixels):
                raise ValueError('Output contract differs')
            cases.append(dict(policy=mode, synthetic_dtype=np.dtype(dtype).name,
                              synthetic_shape=list(shape), exact_source_equality=True,
                              values_compared=int(actual.size), inputs_unchanged=True))
    if before != {str(p.relative_to(ROOT)): sha(p) for p in paths}:
        raise ValueError('Implementation changed')
    return dict(format='osic-image-preprocessing-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, source_revision=REVISION,
        source_function_sha256=FUNCTION_PINS, original_source_version=scope['source_version_id'],
        software_sha256=before, cases=cases,
        runtime=dict(numpy=np.__version__, opencv=cv2.__version__,
                     installed_opencv_distributions={d.metadata['Name']: d.version for d in metadata.distributions()
                         if d.metadata['Name'].lower() in ('opencv-python', 'opencv-python-headless')}),
        resource_policy=dict(native_threads=1, opencv_parallel_workers_disabled=True),
        limitations=['Decoded integer planes only; actual image decoding and metadata types remain unqualified.',
            'Equality uses the current installed OpenCV backend for source and adapter; historical backend parity remains unqualified.',
            'Two OpenCV distributions share a namespace in this environment; package ownership and dependency provenance require resolution before publication.',
            'The different training and inference formulas are intentional source fidelity, not a recommendation for new applications.',
            'No training, full workflow, catalog approval or clinical validity is established.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    result = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_image_preprocessing_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, cases=len(result['cases']),
        values_compared=sum(c['values_compared'] for c in result['cases']), approved=False)))

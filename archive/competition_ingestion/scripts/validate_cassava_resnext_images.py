"""Pinned-source transform comparisons and synthetic OpenCV decode parity."""
import hashlib
import importlib.metadata
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch

from sciona.cassava_resnext_images import decode_rgb, transformations, transform_rgb
from scripts.validate_cassava_source_components import source_helpers


def main():
    source = Path('/private/tmp/sciona_cassava_winner_source/resnext.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['resnext']['code_sha256']
    namespace = {name: getattr(A, name) for name in ['Compose', 'RandomResizedCrop', 'Transpose', 'HorizontalFlip', 'VerticalFlip', 'ShiftScaleRotate', 'Normalize', 'Resize']}
    namespace.update(ToTensorV2=ToTensorV2, CFG=SimpleNamespace(size=512))
    ns = source_helpers(source, {'get_transforms'}, namespace)
    raw = np.random.default_rng(514).integers(0, 256, (600, 800, 3), dtype=np.uint8)
    before = raw.copy()
    for training in (True, False):
        actual = transformations(training=training)
        expected = ns['get_transforms'](data='train' if training else 'valid')
        assert actual.to_dict() == expected.to_dict()
        for seed in range(20):
            random.seed(seed); np.random.seed(seed)
            first = transform_rgb(raw, actual)
            random.seed(seed); np.random.seed(seed)
            second = expected(image=raw)['image']
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    resized = cv2.resize(raw, (512, 512)).astype(np.float64)
    scalar = (resized / 255. - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    np.testing.assert_allclose(transform_rgb(raw, transformations(training=False)).numpy(), scalar.transpose(2, 0, 1), atol=5e-7)
    with tempfile.TemporaryDirectory(prefix='cassava-decoder-diagnostic-') as temp:
        path = Path(temp) / 'synthetic.jpg'
        success, encoded = cv2.imencode('.jpg', cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        assert success
        path.write_bytes(encoded.tobytes())
        np.testing.assert_array_equal(decode_rgb(encoded.tobytes()), cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    np.testing.assert_array_equal(raw, before)
    rejected = 0
    for encoded in (b'', b'invalid synthetic bytes', None):
        try: decode_rgb(encoded)
        except ValueError: rejected += 1
        else: raise AssertionError('Invalid encoding accepted')
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'training_transform_comparisons': 20, 'validation_transform_comparisons': 20,
              'configuration_match': True, 'scalar_normalization_match': True,
              'opencv_decode_match': True, 'input_unchanged': True, 'invalid_encodings_rejected': rejected,
              'runtime': {'albumentations': importlib.metadata.version('albumentations'), 'opencv': cv2.__version__},
              'scope': 'Reference decoding and source transform parity; historical augmentation runtime and full fold lifecycle remain unverified.'}
    paths = ['sciona/cassava_resnext_images.py', 'scripts/validate_cassava_resnext_images.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_resnext_images_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

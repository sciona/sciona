"""Direct pinned-source comparisons on synthetic pixels only."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from sciona.cassava_image_views import efficientnet_tiles, efficientnet_views, cropnet_view
from scripts.validate_cassava_source_components import source_helpers


def main():
    source = Path('/private/tmp/sciona_cassava_winner_source/inference.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['inference']['code_sha256']
    ns = source_helpers(source, {'create_image_tiles', 'image_augmentations', 'augment_tiles_light', 'cut_crop_image'},
                        {'np': np, 'tf': tf, 'cv2': cv2, 'IMAGE_SIZE': (512, 512)})
    raw = np.random.default_rng(74).integers(0, 256, size=(600, 800, 3), dtype=np.uint8)
    before = raw.copy()
    expected_tiles = ns['create_image_tiles']((800, 600), raw.astype(np.float32))
    np.testing.assert_array_equal(efficientnet_tiles(raw), expected_tiles)
    for seed in (3, 19, 41):
        tf.random.set_seed(seed)
        expected = ns['augment_tiles_light'](expected_tiles).numpy()
        tf.random.set_seed(seed)
        actual = efficientnet_views(raw).numpy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[8], expected_tiles[4])
        np.testing.assert_array_equal(actual[9], expected_tiles[4])
    expected_crop = ns['cut_crop_image']((raw / 255.).astype(np.float32))
    np.testing.assert_array_equal(cropnet_view(raw).numpy(), expected_crop)
    np.testing.assert_array_equal(raw, before)
    rejected = 0
    for invalid in (raw[:500], raw[:, :600], raw[:, :, :1], raw.astype(np.float32)):
        for fn in (efficientnet_tiles, efficientnet_views, cropnet_view):
            try:
                fn(invalid)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('Invalid decoded-image contract accepted')
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'tile_comparisons': 5, 'augmented_view_comparisons': 30, 'cropnet_view_comparisons': 1,
              'invalid_contracts_rejected': rejected, 'input_unchanged': True,
              'runtime': {p: importlib.metadata.version(p) for p in ['tensorflow', 'tf-keras', 'numpy']},
              'opencv_version': cv2.__version__,
              'scope': 'Decoded-pixel inference transformations only; no decoder, trained models, original runtime or training augmentation qualification.'}
    paths = ['sciona/cassava_image_views.py', 'scripts/validate_cassava_image_views.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_image_views_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

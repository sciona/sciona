"""Actual licensed public CropNet inference and frozen-variable checks."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
from pathlib import Path

import numpy as np

from sciona.cassava_cropnet import load_model, predict_images
from sciona.cassava_image_views import cropnet_view


def main():
    root = Path('/private/tmp/sciona_cassava_cropnet_reference/model')
    artifact = json.loads(Path('docs/reviews/competition_cassava_cropnet_artifact.json').read_text())
    model = load_model(root, expected_hashes=artifact['files'])
    before = [v.numpy().copy() for v in model.variables]
    rng = np.random.default_rng(135)
    images = [rng.integers(0, 256, (600, 800, 3), dtype=np.uint8), np.full((600, 800, 3), 127, np.uint8)]
    actual = predict_images(model, images)
    for i, image in enumerate(images):
        probabilities = model(cropnet_view(image), training=False).numpy()[0]
        expected = np.array([float(probabilities[j]) + float(probabilities[5]) / 5 for j in range(5)])
        np.testing.assert_allclose(actual[i], expected, atol=1e-8)
    np.testing.assert_array_equal(actual, predict_images(model, images))
    for value, previous in zip(model.variables, before, strict=True): np.testing.assert_array_equal(value.numpy(), previous)
    bad = dict(artifact['files'])
    first = next(iter(bad)); bad[first] = '0' * 64
    try: load_model(root, expected_hashes=bad)
    except ValueError: pass
    else: raise AssertionError('Tampered manifest accepted')
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'official_pretrained_artifact_used': True, 'images_executed': 2,
              'model_variables_unchanged': len(before), 'repeat_predictions_identical': True,
              'unknown_mapping_scalar_match': True, 'tampered_manifest_rejected': True,
              'scope': 'Official licensed CropNet reference artifact; identity with winner cached copy and full ensemble remain unverified.'}
    paths = ['sciona/cassava_cropnet.py', 'sciona/cassava_image_views.py', 'sciona/cassava_ensemble.py',
             'scripts/validate_cassava_cropnet.py', 'docs/reviews/competition_cassava_cropnet_artifact.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_cropnet_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

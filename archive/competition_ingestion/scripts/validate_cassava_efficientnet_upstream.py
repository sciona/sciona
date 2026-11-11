"""Verify upstream reconstruction after source population normalization adapt."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import tensorflow as tf

from sciona.cassava_efficientnet_model import build_model
from sciona.cassava_efficientnet_upstream import build_adapted_model
from sciona.cassava_training_images import adapt_training_normalization


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    base = Path('/private/tmp')
    source_root = base / 'sciona_cassava_efficientnet_winner_reference'
    upstream_root = base / 'sciona_cassava_efficientnet_upstream_reference'
    source_pin = json.loads((source_root / 'download_manifest.json').read_text())
    upstream_pin = json.loads((upstream_root / 'manifest.json').read_text())
    rng = np.random.default_rng(931)
    pixels = [rng.integers(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(2)]
    images = [tf.image.encode_jpeg(image, quality=91).numpy() for image in pixels]
    tf.keras.backend.clear_session()
    reference = build_model(weights=source_root / 'model.h5', expected_sha256=source_pin['model_sha256'])
    tf.random.set_seed(315)
    adapt_training_normalization(reference, images, batch_size=1)
    tf.keras.backend.clear_session()
    tf.random.set_seed(315)
    actual = build_adapted_model(weights=upstream_root / 'model.h5', expected_sha256=upstream_pin['sha256'],
                                 training_jpegs=images, batch_size=1)
    source_backbone = reference.get_layer('efficientnetb4')
    actual_backbone = actual.get_layer('efficientnetb4')
    assert len(source_backbone.weights) == len(actual_backbone.weights) == 611
    for source, target in zip(source_backbone.weights, actual_backbone.weights, strict=True):
        assert source.name == target.name
        np.testing.assert_array_equal(source.numpy(), target.numpy())
    # Both constructions use a fresh random task head. Share it explicitly to
    # isolate backbone/normalization equivalence from random initialization.
    actual.layers[-1].set_weights(reference.layers[-1].get_weights())
    batch = tf.stack([tf.cast(tf.image.decode_jpeg(value, channels=3), tf.float32) for value in images])
    expected = reference(batch, training=False).numpy()
    result = actual(batch, training=False).numpy()
    np.testing.assert_array_equal(result, expected)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(-1), 1., atol=1e-6)
    with tempfile.TemporaryDirectory(prefix='cassava-upstream-negative-') as directory:
        invalid = Path(directory) / 'synthetic.h5'
        with h5py.File(invalid, 'w') as artifact:
            artifact.create_dataset('unexpected', data=np.zeros((1,), dtype=np.float32))
        digest = hashlib.sha256(invalid.read_bytes()).hexdigest()
        try:
            build_adapted_model(weights=invalid, expected_sha256=digest, training_jpegs=images, batch_size=1)
        except ValueError as error:
            assert 'hierarchy' in str(error)
        else:
            raise AssertionError('Invalid model parameter hierarchy accepted')
    paths = ['sciona/cassava_efficientnet_upstream.py', 'sciona/cassava_efficientnet_weights.py',
        'sciona/cassava_efficientnet_model.py', 'sciona/cassava_training_images.py',
        'scripts/validate_cassava_efficientnet_upstream.py',
        'docs/reviews/competition_cassava_efficientnet_weight_origin_validation.json']
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
        'upstream_sha256': upstream_pin['sha256'], 'source_sha256': source_pin['model_sha256'],
        'identical_post_adaptation_backbone_tensors': 611, 'shared_random_task_head': True,
        'full_resolution_images': 2, 'post_adaptation_predictions_exactly_equal': True,
        'malformed_parameter_hierarchy_rejected': True,
        'scope': 'Numerical equivalence after explicit training-population adaptation; upstream redistribution terms, full pretrained lifecycle and complete ensemble remain separate gates.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_efficientnet_upstream_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS upstream reconstruction: all 611 adapted backbone tensors and full-resolution predictions identical', flush=True)


if __name__ == '__main__':
    main()

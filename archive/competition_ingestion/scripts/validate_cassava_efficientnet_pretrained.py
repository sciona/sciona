"""Private inspection of exact notebook-referenced weights; no license approval."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sciona.cassava_efficientnet_model import build_model
from sciona.cassava_efficientnet_training import keras_focal_loss


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    root = Path('/private/tmp/sciona_cassava_efficientnet_winner_reference')
    manifest = json.loads((root / 'download_manifest.json').read_text())
    path = root / 'model.h5'
    assert path.stat().st_size == manifest['model_bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['model_sha256']
    model = build_model(weights=str(path), expected_sha256=manifest['model_sha256'])
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    bn = [layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    before_bn = [[value.copy() for value in layer.get_weights()] for layer in bn]
    conv = next(layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.Conv2D))
    before_conv = conv.kernel.numpy().copy()
    before_head = model.layers[-1].kernel.numpy().copy()
    images = tf.random.stateless_uniform((1, 512, 512, 3), seed=(41, 57), maxval=255.)
    targets = tf.one_hot([2], 5)
    with tf.GradientTape() as tape:
        probabilities = model(images, training=True)
        objective = tf.reduce_mean(keras_focal_loss(targets, probabilities))
    gradients = tape.gradient(objective, model.trainable_variables)
    assert np.isfinite(objective.numpy())
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    tf.keras.optimizers.legacy.Adam(learning_rate=1e-6).apply_gradients(zip(gradients, model.trainable_variables))
    assert not np.array_equal(before_conv, conv.kernel.numpy())
    assert not np.array_equal(before_head, model.layers[-1].kernel.numpy())
    for layer, before in zip(bn, before_bn, strict=True):
        assert not layer.trainable
        for actual, expected in zip(layer.get_weights(), before, strict=True):
            np.testing.assert_array_equal(actual, expected)
    actual = model(images, training=False).numpy()
    assert actual.shape == (1, 5) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(-1), 1., atol=1e-6)
    paths = ['sciona/cassava_efficientnet_initialization.py', 'sciona/cassava_efficientnet_upstream.py', 'sciona/cassava_efficientnet_weights.py', 'scripts/validate_cassava_efficientnet_pretrained.py',
        'sciona/cassava_efficientnet_model.py', 'sciona/cassava_efficientnet_training.py']
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
        'artifact': {key: manifest[key] for key in ['url', 'sha256', 'bytes', 'model_sha256', 'model_bytes', 'publisher_license']},
        'strict_weight_load': True, 'full_resolution_training_steps': 1,
        'backbone_and_head_updated': True, 'frozen_batchnorm_layers_unchanged': len(bn),
        'finite_gradients': True, 'post_update_probabilities_valid': True,
        'scope': 'Private compatibility inspection of exact notebook-referenced artifact. Publisher license is Unknown; redistribution is not approved. Population adaptation, complete pretrained lifecycle and ensemble remain separate gates.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_efficientnet_pretrained_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS exact EfficientNet source weights: strict load, full-resolution update, frozen BatchNorm', flush=True)


if __name__ == '__main__':
    main()

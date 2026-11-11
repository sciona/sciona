"""Actual full B4 topology, inference parity and trainability on synthetic input."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sciona.cassava_efficientnet_model import build_model
from scripts.validate_cassava_source_components import source_helpers


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    source = Path('/private/tmp/sciona_cassava_winner_source/efficientnet.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['efficientnet']['code_sha256']
    ns = source_helpers(source, {'efficientnetb4', 'sigmoid_focal_crossentropy'}, {
        'tf': tf, 'keras': keras, 'K': keras.backend, 'EfficientNetB4': keras.applications.EfficientNetB4,
        'img_size': (512, 512)})
    keras.utils.set_random_seed(73)
    model = build_model(weights=None)
    reference = ns['efficientnetb4'](weights=None)
    reference.set_weights(model.get_weights())
    backbone = model.get_layer('efficientnetb4')
    reference_backbone = reference.get_layer('efficientnetb4')
    assert backbone.count_params() == reference_backbone.count_params()
    for a, b in zip(backbone.layers, reference_backbone.layers, strict=True):
        assert type(a) is type(b) and a.get_config() == b.get_config()
        assert a.trainable == b.trainable
    batch = tf.reshape(tf.linspace(0., 255., 512 * 512 * 3), (1, 512, 512, 3))
    np.testing.assert_allclose(model(batch, training=False), reference(batch, training=False), atol=1e-7, rtol=1e-6)
    del reference
    batchnorm = [x for x in backbone.layers if isinstance(x, keras.layers.BatchNormalization)]
    assert batchnorm and all(not x.trainable for x in batchnorm)
    before_bn = [[v.numpy().copy() for v in x.weights] for x in batchnorm]
    head = model.layers[-1]
    before_head = head.kernel.numpy().copy()
    convolution = next(x for x in backbone.layers if isinstance(x, keras.layers.Conv2D))
    before_conv = convolution.kernel.numpy().copy()
    with tf.GradientTape() as tape:
        probabilities = model(batch, training=True)
        loss = tf.reduce_mean(ns['sigmoid_focal_crossentropy'](tf.constant([[0., 1., 0., 0., 0.]]), probabilities))
    gradients = tape.gradient(loss, model.trainable_weights)
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
    keras.optimizers.legacy.SGD(.01).apply_gradients(zip(gradients, model.trainable_weights))
    assert not np.array_equal(before_head, head.kernel.numpy())
    assert not np.array_equal(before_conv, convolution.kernel.numpy())
    for layer, before in zip(batchnorm, before_bn):
        for value, old in zip(layer.weights, before): np.testing.assert_array_equal(value.numpy(), old)
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'pretrained_weights_used': False, 'backbone_parameters': backbone.count_params(),
              'total_parameters': model.count_params(), 'batchnorm_layers_frozen': len(batchnorm),
              'source_inference_match': True, 'full_resolution_training_step': True,
              'head_and_backbone_updated': True, 'batchnorm_weights_unchanged': True,
              'scope': 'Untrained topology diagnostic in reference runtime; no pretrained provenance, population adaptation or complete lifecycle qualification.'}
    paths = ['sciona/cassava_efficientnet_weights.py', 'sciona/cassava_efficientnet_model.py', 'scripts/validate_cassava_efficientnet_model.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_efficientnet_model_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

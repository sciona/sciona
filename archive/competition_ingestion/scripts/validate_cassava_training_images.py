"""Synthetic encoded-image and augmentation comparisons with pinned source."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf

from sciona.cassava_training_images import decode_training_jpeg, augment_training_image, normalization_pixels, adapt_training_normalization
from sciona.cassava_efficientnet_model import build_model
from scripts.validate_cassava_source_components import source_helpers


def main():
    source = Path('/private/tmp/sciona_cassava_winner_source/efficientnet.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['efficientnet']['code_sha256']
    ns = source_helpers(source, {'decode_image', 'image_augmentations', 'get_normalization_batch'}, {'tf': tf, 'partial': partial, 'AUTOTUNE': tf.data.AUTOTUNE, 'default_img_size': (512, 512), 'img_size': (512, 512)})
    ns['get_image_and_label'] = lambda value, train, img_size: (ns['decode_image'](value, img_size), tf.constant(0))
    pixels = np.random.default_rng(294).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    encoded = tf.image.encode_jpeg(pixels, quality=91)
    actual = decode_training_jpeg(encoded)
    expected = ns['decode_image'](encoded, (512, 512))
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    for seed in range(20):
        tf.random.set_seed(seed)
        expected_aug, _ = ns['image_augmentations'](expected, tf.constant(0))
        tf.random.set_seed(seed)
        actual_aug = augment_training_image(actual)
        np.testing.assert_array_equal(actual_aug.numpy(), expected_aug.numpy())
    normalized = normalization_pixels(actual)
    assert normalized.dtype == tf.bfloat16
    np.testing.assert_array_equal(normalized.numpy(), (expected / 255.).numpy())
    # Show that early float32 conversion would change normalization population.
    assert not np.array_equal(tf.cast(normalized, tf.float32).numpy(), (tf.cast(expected, tf.float32) / 255.).numpy())
    model = build_model(weights=None)
    # Independent channel statistics after the exact source decoding/precision path.
    second = tf.image.encode_jpeg(np.full_like(pixels, 73), quality=91)
    reference_stream = ns['get_normalization_batch'](tf.data.Dataset.from_tensor_slices([encoded.numpy(), second.numpy()]), 1)
    values = np.concatenate([tf.cast(x, tf.float32).numpy() for x in reference_stream]).astype(np.float64)
    adapt_training_normalization(model, [encoded.numpy(), second.numpy()], batch_size=1)
    layer = model.get_layer('efficientnetb4').get_layer('normalization')
    reference_layer = tf.keras.layers.Normalization(axis=-1)
    reference_layer.adapt(reference_stream)
    np.testing.assert_allclose(layer.mean.numpy(), reference_layer.mean.numpy(), atol=1e-7)
    np.testing.assert_allclose(layer.variance.numpy(), reference_layer.variance.numpy(), atol=1e-7)
    mean_error = float(np.max(np.abs(layer.mean.numpy().reshape(-1) - values.mean(axis=(0, 1, 2)))))
    variance_error = float(np.max(np.abs(layer.variance.numpy().reshape(-1) - values.var(axis=(0, 1, 2)))))
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'jpeg_decode_comparisons': 1, 'augmentation_comparisons': 20,
              'normalization_preserves_bfloat16': True,
              'actual_backbone_normalization_adapted': True,
              'source_population_adaptation_match': True,
              'float64_population_mean_max_difference': mean_error,
              'float64_population_variance_max_difference': variance_error,
              'numerical_limit': 'Source Keras float32 moments retained; not replaced with float64 statistics.',
              'scope': 'Reference training pixel operations and population moments only; fold selection and complete lifecycle remain unverified.'}
    paths = ['sciona/cassava_efficientnet_weights.py', 'sciona/cassava_training_images.py', 'sciona/cassava_efficientnet_model.py', 'scripts/validate_cassava_training_images.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_training_images_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

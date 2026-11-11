"""Full 14-epoch B4 refit and source-style ten-view prediction on synthetic data."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sciona.cassava_efficientnet_refit import train_final, predict_images
from sciona.cassava_efficientnet_schedule import learning_rate
from scripts.validate_cassava_source_components import source_helpers


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    keras.utils.set_random_seed(745)
    encoded = [tf.image.encode_jpeg(np.full((512, 512, 3), value, np.uint8)).numpy() for value in (41, 89)]
    raw = np.random.default_rng(338).integers(0, 256, (600, 800, 3), dtype=np.uint8)
    source = Path('/private/tmp/sciona_cassava_winner_source/inference.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['inference']['code_sha256']
    ns = source_helpers(source, {'create_image_tiles', 'image_augmentations', 'augment_tiles_light', 'multi_predict_keras'},
        {'tf': tf, 'np': np, 'cv2': cv2, 'IMAGE_SIZE': (512, 512),
         'read_preprocess_file': lambda unused: ((800, 600), raw.astype(np.float32))})
    with tempfile.TemporaryDirectory(prefix='cassava-refit-diagnostic-') as temp:
        model, history, artifact = train_final(encoded, [1, 1], weights=None, batch_size=1,
                                                output_directory=Path(temp) / 'model')
        assert len(history['loss']) == 14 and not any(k.startswith('val_') for k in history)
        assert np.isfinite(history['loss']).all()
        assert int(model.optimizer.iterations.numpy()) == 28
        np.testing.assert_allclose(history['lr'], [learning_rate(i) for i in range(14)], rtol=1e-6)
        loaded = keras.models.load_model(artifact, compile=False)
        for a, b in zip(model.get_weights(), loaded.get_weights(), strict=True): np.testing.assert_array_equal(a, b)
        tf.random.set_seed(842)
        actual = predict_images(loaded, [raw])
        tf.random.set_seed(842)
        expected = ns['multi_predict_keras']('synthetic', loaded)
        np.testing.assert_allclose(actual[0], expected, atol=1e-6, rtol=1e-5)
        report = {'approved': False, 'passed': True, 'synthetic_only': True,
                  'epochs_executed': 14, 'optimizer_steps': 28, 'diagnostic_batch_size': 1,
                  'pretrained_weights_used': False, 'checkpoint_roundtrip': True,
                  'source_ten_view_prediction_match': True,
                  'explicit_reconstruction': 'Final normalization adapts the supplied full training population; source commented refit block has an unresolved scratch variable.',
                  'scope': 'Full-resolution CPU refit diagnostic with complete epoch budget; no pretrained, original batch-size, TPU or full competition graph qualification.'}
    paths = ['sciona/cassava_efficientnet_initialization.py', 'sciona/cassava_efficientnet_upstream.py', 'sciona/cassava_efficientnet_weights.py', 'sciona/cassava_efficientnet_refit.py', 'sciona/cassava_efficientnet_training.py',
             'sciona/cassava_efficientnet_model.py', 'sciona/cassava_efficientnet_schedule.py',
             'sciona/cassava_training_images.py', 'sciona/cassava_training_stream.py', 'sciona/cassava_image_views.py',
             'scripts/validate_cassava_efficientnet_refit.py', 'scripts/validate_cassava_source_components.py',
             'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_efficientnet_refit_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

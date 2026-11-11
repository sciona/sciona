"""Actual pretrained upstream fold and fourteen-epoch final refit."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sciona.cassava_efficientnet_training import train_fold, keras_focal_loss
from sciona.cassava_efficientnet_refit import train_final, predict_images
from sciona.cassava_efficientnet_schedule import learning_rate


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(515)
    root = Path('/private/tmp/sciona_cassava_efficientnet_upstream_reference')
    pin = json.loads((root / 'manifest.json').read_text())
    settings = dict(weights=root / 'model.h5', expected_sha256=pin['sha256'], weight_format='upstream', batch_size=1)
    rng = np.random.default_rng(951)
    def jpeg():
        return tf.image.encode_jpeg(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), quality=91).numpy()
    training, validation = (jpeg(), jpeg()), (jpeg(),)
    results = {}
    with tempfile.TemporaryDirectory(prefix='cassava-upstream-lifecycle-') as directory:
        print('START pretrained EfficientNet fold', flush=True)
        model, history, checkpoint = train_fold(training, (1, 3), validation, (1,),
            checkpoint_directory=Path(directory) / 'fold', **settings)
        epochs = len(history['loss'])
        assert 1 <= epochs <= 20
        assert all(np.isfinite(history[key]).all() for key in ['loss', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy'])
        np.testing.assert_allclose(history['lr'], [learning_rate(i) for i in range(epochs)], rtol=1e-6)
        saved = keras.models.load_model(checkpoint, custom_objects={'keras_focal_loss': keras_focal_loss})
        best = int(np.argmax(history['val_categorical_accuracy']))
        assert int(saved.optimizer.iterations.numpy()) == 2 * (best + 1)
        if epochs < 20:
            for value, expected in zip(model.get_weights(), saved.get_weights(), strict=True):
                np.testing.assert_array_equal(value, expected)
        results['fold'] = {'epochs_executed': epochs, 'maximum_epochs': 20,
            'best_epoch': best, 'checkpoint_optimizer_step_verified': True,
            'early_stopped': epochs < 20, 'restored_weights_verified': epochs < 20,
            'learning_rate_history_verified': True}
        del model, saved
        keras.backend.clear_session()
        gc.collect()
        print('PASS pretrained fold; START full fourteen-epoch pretrained refit', flush=True)
        model, history, checkpoint = train_final(training, (1, 3),
            output_directory=Path(directory) / 'final', **settings)
        assert len(history['loss']) == 14 and np.isfinite(history['loss']).all()
        np.testing.assert_allclose(history['lr'], [learning_rate(i) for i in range(14)], rtol=1e-6)
        assert int(model.optimizer.iterations.numpy()) == 28
        saved = keras.models.load_model(checkpoint, custom_objects={'keras_focal_loss': keras_focal_loss})
        for value, expected in zip(model.get_weights(), saved.get_weights(), strict=True):
            np.testing.assert_array_equal(value, expected)
        query = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
        tf.random.set_seed(543)
        actual = predict_images(model, [query])
        tf.random.set_seed(543)
        expected = predict_images(saved, [query])
        np.testing.assert_array_equal(actual, expected)
        results['refit'] = {'epochs_executed': 14, 'optimizer_steps': 28,
            'learning_rate_history_verified': True, 'all_checkpoint_weights_identical': True,
            'ten_view_prediction_roundtrip_exact': True}
    paths = ['sciona/cassava_efficientnet_initialization.py', 'sciona/cassava_efficientnet_upstream.py',
        'sciona/cassava_efficientnet_weights.py', 'sciona/cassava_efficientnet_model.py',
        'sciona/cassava_efficientnet_training.py', 'sciona/cassava_efficientnet_refit.py',
        'sciona/cassava_efficientnet_schedule.py', 'sciona/cassava_training_images.py',
        'sciona/cassava_training_stream.py', 'sciona/cassava_image_views.py',
        'scripts/validate_cassava_efficientnet_upstream_lifecycle.py']
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
        'upstream_pretrained_sha256': pin['sha256'], 'diagnostic_batch_size': 1,
        'results': results, 'temporary_checkpoints_removed': True,
        'scope': 'Actual upstream pretrained CPU fold and final refit with source epoch/callback mechanics. Complete fivefold CV, original population/budget and full ensemble remain unqualified.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_efficientnet_upstream_lifecycle_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS complete pretrained fold and fourteen-epoch refit', flush=True)


if __name__ == '__main__':
    main()

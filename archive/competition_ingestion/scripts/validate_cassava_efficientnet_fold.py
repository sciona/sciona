"""Execute the real B4 fold lifecycle with synthetic, explicitly untrained inputs."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sciona.cassava_efficientnet_schedule import learning_rate
from sciona.cassava_efficientnet_training import train_fold, keras_focal_loss


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    keras.utils.set_random_seed(432)
    def jpeg(value):
        return tf.image.encode_jpeg(np.full((512, 512, 3), value, np.uint8)).numpy()
    training = [jpeg(41), jpeg(89)]
    validation = [jpeg(127)]
    with tempfile.TemporaryDirectory(prefix='cassava-fold-diagnostic-') as temp:
        model, history, checkpoint = train_fold(training, [1, 1], validation, [4], weights=None,
                                                 batch_size=1, checkpoint_directory=Path(temp) / 'checkpoint')
        epochs = len(history['loss'])
        assert 1 <= epochs <= 20 and checkpoint.is_file()
        assert all(np.isfinite(history[key]).all() for key in ['loss', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy'])
        np.testing.assert_allclose(history['lr'], [learning_rate(i) for i in range(epochs)], rtol=1e-6)
        saved = keras.models.load_model(checkpoint, custom_objects={'keras_focal_loss': keras_focal_loss})
        best_epoch = int(np.argmax(history['val_categorical_accuracy']))
        assert int(saved.optimizer.iterations.numpy()) == (best_epoch + 1) * 2
        # Early-stop restoration should match the source's saved first best model.
        if epochs < 20:
            for actual, expected in zip(model.get_weights(), saved.get_weights(), strict=True):
                np.testing.assert_array_equal(actual, expected)
        report = {'approved': False, 'passed': True, 'synthetic_only': True,
                  'pretrained_weights_used': False, 'diagnostic_batch_size': 1,
                  'epochs_executed': epochs, 'maximum_epochs': 20,
                  'checkpoint_best_epoch': best_epoch, 'checkpoint_roundtrip': True,
                  'early_stopped': epochs < 20, 'restored_weights_verified': epochs < 20,
                  'scope': 'Actual full-resolution B4 CPU lifecycle diagnostic; no pretrained weights, original batch size, fold provenance or TPU parity qualification.'}
    paths = ['sciona/cassava_efficientnet_initialization.py', 'sciona/cassava_efficientnet_upstream.py', 'sciona/cassava_efficientnet_weights.py', 'sciona/cassava_efficientnet_training.py', 'sciona/cassava_efficientnet_model.py',
             'sciona/cassava_efficientnet_schedule.py', 'sciona/cassava_training_images.py',
             'sciona/cassava_training_stream.py', 'scripts/validate_cassava_efficientnet_fold.py']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_efficientnet_fold_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

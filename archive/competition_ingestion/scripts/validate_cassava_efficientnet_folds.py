"""Execute complete pretrained fivefold CV through shared membership dispatch."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sciona.cassava_efficientnet_folds import train_five_folds
from sciona.cassava_efficientnet_training import keras_focal_loss
from sciona.cassava_efficientnet_schedule import learning_rate
from sciona.cassava_fold_contract import build_plan, members


def main():
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    keras.utils.set_random_seed(819)
    root = Path('/private/tmp/sciona_cassava_efficientnet_upstream_reference')
    pin = json.loads((root / 'manifest.json').read_text())
    images = [tf.image.encode_jpeg(np.full((512, 512, 3), 70 + i, np.uint8)).numpy() for i in range(25)]
    assert len(set(images)) == 25
    plan = build_plan([f'synthetic-{i}' for i in range(25)],
        [i % 5 for i in range(25)], [i // 5 for i in range(25)])
    results = []
    with tempfile.TemporaryDirectory(prefix='cassava-efficientnet-fivefold-') as directory:
        print('START five complete upstream pretrained EfficientNet folds', flush=True)
        histories, scores, checkpoints = train_five_folds(plan, images,
            weights=root / 'model.h5', expected_sha256=pin['sha256'], batch_size=1,
            output_directory=Path(directory) / 'cv')
        for fold in range(5):
            train, valid = members(plan, fold)
            assert len(train) == 20 and len(valid) == 5 and not set(train) & set(valid)
            history = histories[fold]
            epochs = len(history['loss'])
            assert 1 <= epochs <= 20
            assert all(np.isfinite(values).all() for values in history.values())
            np.testing.assert_allclose(history['lr'], [learning_rate(i) for i in range(epochs)], rtol=1e-6)
            best = int(np.argmax(history['val_categorical_accuracy']))
            np.testing.assert_allclose(scores[fold]['categorical_accuracy'], history['val_categorical_accuracy'][best], atol=1e-7)
            np.testing.assert_allclose(scores[fold]['loss'], history['val_loss'][best], rtol=1e-5, atol=1e-6)
            saved = keras.models.load_model(checkpoints[fold], custom_objects={'keras_focal_loss': keras_focal_loss})
            assert int(saved.optimizer.iterations.numpy()) == (best + 1) * 20
            results.append({'fold': fold, 'epochs_executed': epochs, 'best_epoch': best,
                'held_out_checkpoint_score_replayed': True, 'checkpoint_optimizer_count_verified': True})
            del saved
            keras.backend.clear_session()
    paths = ['sciona/cassava_efficientnet_folds.py', 'sciona/cassava_fold_contract.py',
        'sciona/cassava_fold_dispatch.py', 'sciona/cassava_efficientnet_initialization.py',
        'sciona/cassava_efficientnet_upstream.py', 'sciona/cassava_efficientnet_weights.py',
        'sciona/cassava_efficientnet_model.py', 'sciona/cassava_efficientnet_training.py',
        'sciona/cassava_training_images.py', 'sciona/cassava_training_stream.py',
        'sciona/cassava_efficientnet_schedule.py', 'scripts/validate_cassava_efficientnet_folds.py']
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
        'upstream_pretrained_sha256': pin['sha256'], 'diagnostic_batch_size': 1,
        'folds': results, 'temporary_checkpoints_removed': True,
        'scope': 'Complete pretrained fivefold CPU CV through shared population dispatch; original budgets, preparation provenance and full four-family ensemble remain separate gates.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_efficientnet_folds_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS all five pretrained EfficientNet folds and selected checkpoint evaluations', flush=True)


if __name__ == '__main__':
    main()

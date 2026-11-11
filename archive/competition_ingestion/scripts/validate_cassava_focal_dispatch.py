"""Probe pinned source loss in an isolated TensorFlow/legacy-Keras runtime.

Run with the private reference package directory on PYTHONPATH. This is a
modern reference runtime, not a claim of historical Docker-image parity.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch

from sciona.cassava_focal import focal_loss
from scripts.validate_cassava_source_components import source_helpers


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    source = Path('/private/tmp/sciona_cassava_winner_source/efficientnet.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['efficientnet']['code_sha256']
    fn = source_helpers(source, {'sigmoid_focal_crossentropy'}, {'tf': tf, 'K': keras.backend})['sigmoid_focal_crossentropy']
    rng = np.random.default_rng(51)
    rows = rng.normal(size=(8, 5)).astype('float32')
    labels = np.eye(5, dtype='float32')[np.arange(8) % 5]
    observations = []
    for eager in (True, False):
        model = keras.Sequential([keras.layers.Input(shape=(5,)), keras.layers.Dense(5, activation='softmax', use_bias=False)])
        model.layers[-1].set_weights([np.eye(5, dtype='float32')])
        cached = []

        def loss(y, p):
            cached.append(hasattr(tf.convert_to_tensor(p), '_keras_logits'))
            return fn(y, p)

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=.01), loss=loss, run_eagerly=eager)
        observed_loss = float(model.train_on_batch(rows, labels))
        weight = torch.eye(5, requires_grad=True)
        expected = focal_loss(torch.tensor(rows) @ weight, torch.tensor(labels), bce_branch='cached_logits').mean()
        expected.backward()
        np.testing.assert_allclose(observed_loss, expected.detach().numpy(), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(model.layers[-1].get_weights()[0], (weight - .01 * weight.grad).detach().numpy(), rtol=2e-5, atol=2e-6)
        assert cached and all(cached)
        observations.append({'eager': eager, 'cached_logits_retained': True, 'source_loss_and_sgd_update_match': True})
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'runtime': {p: importlib.metadata.version(p) for p in ['tensorflow', 'tf-keras', 'numpy']},
              'observations': observations,
              'historical_image_equivalence_established': False,
              'scope': 'Compiled Dense-softmax source-loss dispatch in modern legacy Keras; no backbone or original TPU runtime qualification.'}
    paths = ['scripts/validate_cassava_focal_dispatch.py', 'sciona/cassava_focal.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_focal_dispatch_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

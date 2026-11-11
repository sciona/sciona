"""Pipeline structure and actual finite validation batches using synthetic JPEGs."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf

from sciona.cassava_training_stream import training_stream, validation_stream, epoch_steps, prepare_record
from scripts.validate_cassava_source_components import source_helpers


def operations(dataset):
    graph = tf.compat.v1.GraphDef()
    graph.ParseFromString(dataset._as_serialized_graph().numpy())
    return [n.op for n in graph.node if 'Dataset' in n.op]


def main():
    source = Path('/private/tmp/sciona_cassava_winner_source/efficientnet.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['efficientnet']['code_sha256']
    ns = source_helpers(source, {'get_training_dataset', 'get_validation_dataset'}, {
        'tf': tf, 'partial': partial, 'AUTOTUNE': tf.data.AUTOTUNE, 'default_img_size': (512, 512),
        'get_image_and_label': lambda pair, train, img_size: prepare_record(pair[0], pair[1]),
        'image_augmentations': lambda x, y: (x, y)})
    encoded = [tf.image.encode_jpeg(np.full((512, 512, 3), value, np.uint8)).numpy() for value in (31, 73, 149)]
    records = tf.data.Dataset.from_tensor_slices((encoded, [0, 3, 4]))
    # The source parser takes one record; pack the synthetic pair to retain its
    # calling convention. Its image augmentation is already separately verified.
    packed = records.map(lambda x, y: {'encoded': x, 'label': y})
    ns['get_image_and_label'] = lambda pair, train, img_size: prepare_record(pair['encoded'], pair['label'])
    source_train = ns['get_training_dataset'](packed, 2)
    actual_train = training_stream(records, batch_size=2)
    source_ops = operations(source_train)[2:]  # omit source synthetic packing map
    actual_ops = operations(actual_train)[1:]
    assert actual_ops == source_ops, (actual_ops, source_ops)
    assert actual_ops == ['MapDataset', 'RepeatDataset', 'MapDataset', 'ShuffleDatasetV3', 'BatchDatasetV2', 'PrefetchDataset']
    graph = tf.compat.v1.GraphDef()
    graph.ParseFromString(actual_train._as_serialized_graph().numpy())
    nodes = {n.name: n for n in graph.node}
    shuffle = next(n for n in graph.node if n.op == 'ShuffleDatasetV3')
    assert int(tf.make_ndarray(nodes[shuffle.input[1]].attr['value'].tensor)) == 1000
    actual = list(validation_stream(records, batch_size=2))
    expected = list(ns['get_validation_dataset'](packed, 2))
    assert [int(x.shape[0]) for x, _ in actual] == [2, 1]
    for (x, y), (a, b) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(x.numpy(), a.numpy())
        np.testing.assert_array_equal(y.numpy(), b.numpy())
    training_batches = 0
    seen = set()
    for images, labels in actual_train.take(20):
        assert images.shape == (2, 512, 512, 3) and images.dtype == tf.float32
        assert bool(tf.reduce_all(tf.math.is_finite(images)))
        np.testing.assert_array_equal(tf.reduce_sum(labels, axis=-1).numpy(), [1., 1.])
        seen.update(tf.argmax(labels, axis=-1).numpy().tolist())
        training_batches += 1
    assert seen == {0, 3, 4} and training_batches == 20
    assert epoch_steps(3, 2) == 1 and epoch_steps(4, 2) == 2
    rejected = 0
    for size, batch in [(0, 1), (1, 2), (3, 0), (True, 1), (3, True)]:
        try: epoch_steps(size, batch)
        except ValueError: rejected += 1
        else: raise AssertionError('Invalid epoch accepted')
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'training_dataset_ops': actual_ops, 'shuffle_buffer': 1000,
              'validation_batches_compared': 2, 'validation_remainder_retained': True,
              'invalid_epoch_contracts_rejected': rejected, 'training_stream_consumed': True,
              'full_buffer_training_batches': training_batches,
              'scope': 'Training operation graph, full-buffer consumption and finite validation parity; full training lifecycle and original runtime remain unverified.'}
    paths = ['sciona/cassava_training_stream.py', 'sciona/cassava_training_images.py',
             'scripts/validate_cassava_training_stream.py', 'scripts/validate_cassava_source_components.py',
             'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_training_stream_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

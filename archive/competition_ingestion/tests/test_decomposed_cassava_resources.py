"""Real TensorFlow data-policy checks using only synthetic scalar records."""
import pytest


def test_bounded_dataset_preserves_order_batches_and_restores_on_error():
    # Isolate process-wide framework configuration from other tests.
    import os
    import subprocess
    import sys
    environment = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        environment[name] = '1'
    environment['TF_USE_LEGACY_KERAS'] = '1'
    result = subprocess.run([sys.executable, __file__], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def check_policy():
    from sciona.decomposed_cassava_resources import bounded_execution
    import tensorflow as tf
    original = tf.data.Dataset.prefetch
    with pytest.raises(RuntimeError, match='synthetic failure'):
        with bounded_execution() as policy:
            import cv2
            import torch
            from threadpoolctl import threadpool_info
            assert policy['reserved_cores'] == 4
            assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
            assert cv2.getNumThreads() == 1
            assert tf.config.threading.get_intra_op_parallelism_threads() == 1
            assert tf.config.threading.get_inter_op_parallelism_threads() == 1
            assert all(p['num_threads'] == 1 for p in threadpool_info())
            stream = (tf.data.Dataset.range(7).map(lambda x: x * 3)
                      .repeat(2).batch(4).prefetch(tf.data.AUTOTUNE))
            options = stream.options()
            assert options.threading.private_threadpool_size == 1
            assert options.threading.max_intra_op_parallelism == 1
            assert options.autotune.enabled is False
            assert options.experimental_optimization.map_parallelization is False
            assert options.experimental_optimization.apply_default_optimizations is False
            assert [v.tolist() for v in stream.as_numpy_iterator()] == [
                [0, 3, 6, 9], [12, 15, 18, 0], [3, 6, 9, 12], [15, 18]]
            from sciona.cassava_training_stream import training_stream, validation_stream
            from sciona.cassava_training_images import adapt_training_normalization
            import numpy as np
            encoded = tf.io.encode_jpeg(tf.fill([512, 512, 3], tf.constant(127, tf.uint8))).numpy()
            records = tf.data.Dataset.from_tensor_slices(([encoded, encoded], [0, 1]))
            training = training_stream(records, batch_size=1)
            validation = validation_stream(records, batch_size=1)
            for dataset in (training, validation):
                assert dataset.options().threading.private_threadpool_size == 1
                assert dataset.options().autotune.enabled is False
            # Do not consume the infinite training shuffle buffer here.
            batches = list(validation.as_numpy_iterator())
            assert len(batches) == 2
            np.testing.assert_array_equal(batches[0][1], [[1, 0, 0, 0, 0]])
            np.testing.assert_array_equal(batches[1][1], [[0, 1, 0, 0, 0]])
            class NormalizationCapture:
                def get_layer(self, name):
                    return self

                def adapt(self, dataset):
                    assert dataset.options().threading.private_threadpool_size == 1
                    assert dataset.options().autotune.enabled is False
                    pixels = list(dataset.as_numpy_iterator())
                    assert len(pixels) == 2
                    expected = (tf.cast(127, tf.bfloat16) / 255.).numpy()
                    assert all(np.all(batch == expected) for batch in pixels)
            adapt_training_normalization(NormalizationCapture(), [encoded, encoded], batch_size=1)
            raise RuntimeError('synthetic failure')
    assert tf.data.Dataset.prefetch is original


if __name__ == '__main__':
    check_policy()

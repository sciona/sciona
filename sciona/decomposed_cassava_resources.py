"""Explicit resource policy for serial Cassava qualification processes.

Reserve the entire four-core task budget while this context runs. No concurrent
agent compute, Torch data-loader workers, or extra dataset iterators are allowed.
Training and validation each have one private data worker; native kernels and
framework dispatch are single threaded. Apply before framework initialization.
This changes scheduling/prefetch only, not training populations or fit budgets.
It does not establish identical random augmentation realizations across schedules.
"""
from contextlib import contextmanager
import os
from unittest.mock import patch


@contextmanager
def bounded_execution():
    """Configure a dedicated process and bound every Dataset.prefetch call.

    Framework settings remain in effect after exit; threadpoolctl restores its
    prior limits. Dataset replacement is scoped and restored even on failure.
    Use only in a dedicated qualification
process; no concurrent callers may create datasets outside this policy.
"""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[name] = '1'
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    import cv2
    import tensorflow as tf
    import torch
    from threadpoolctl import threadpool_limits

    # This pinned macOS build ignores a limit of one (still reports 16).
    # Zero disables OpenCV internal parallelism and reports one thread.
    cv2.setNumThreads(0)
    if cv2.getNumThreads() != 1:
        raise RuntimeError('OpenCV internal parallelism could not be disabled')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    original = tf.data.Dataset.prefetch

    def prefetch(dataset, buffer_size, name=None):
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        options.autotune.enabled = False
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.map_parallelization = False
        return original(dataset, 1, name=name).with_options(options)

    with threadpool_limits(limits=1), patch.object(tf.data.Dataset, 'prefetch', prefetch):
        yield dict(reserved_cores=4, native_threads=1, torch_workers=0,
                   tf_data_private_threads=1, prefetch_elements=1,
                   autotune=False, maximum_concurrent_data_iterators=2)

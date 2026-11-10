"""Native legacy-Keras prediction with explicit batching and loader limits."""
import numpy as np


def make_predictor(model, *, batch_size):
    """Create a predictor over aligned finite floating input arrays.

Inputs are a nonempty list/tuple of float32/64 arrays with a shared nonzero
leading dimension. One tensor output with that same leading dimension is
required. Native Keras array conversion, prediction graph and batch ordering
are retained, including the final partial batch. The private dataset pool has
one worker, with autotuning disabled. Callers reserve that worker in addition
to their explicitly bounded model-compute pool. No fitting occurs here.
"""
    import tensorflow as tf
    import tf_keras
    from tf_keras.src.engine.data_adapter import TensorLikeDataAdapter
    if not isinstance(model,tf_keras.Model) or type(batch_size) is not int or batch_size<=0:
        raise ValueError('Legacy-Keras model and positive integer batch size required')

    def predict(inputs):
        if not isinstance(inputs,(list,tuple)) or not inputs:
            raise ValueError('Explicit aligned input arrays required')
        for value in inputs:
            if (not isinstance(value,np.ndarray) or value.ndim<1 or not len(value)
                    or value.dtype not in (np.dtype('float32'),np.dtype('float64'))
                    or not np.isfinite(value).all()):
                raise ValueError('Nonempty finite floating input arrays required')
        size=len(inputs[0])
        if any(len(value)!=size for value in inputs):
            raise ValueError('Aligned input row counts required')
        adapter=TensorLikeDataAdapter(list(inputs),batch_size=batch_size,shuffle=False)
        options=tf.data.Options()
        options.threading.private_threadpool_size=1
        options.threading.max_intra_op_parallelism=1
        options.autotune.enabled=False
        dataset=adapter.get_dataset().with_options(options)
        value=model.predict(dataset,verbose=0)
        if (not isinstance(value,np.ndarray) or value.ndim<1 or len(value)!=size
                or value.dtype.kind!='f' or not np.isfinite(value).all()):
            raise ValueError('Finite aligned floating predictions required')
        return value

    return predict

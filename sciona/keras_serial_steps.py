"""Use compiled legacy-Keras steps without a parallel dataset loader."""


def make_steps(model):
    """Return train/test functions retaining Keras loss aggregation and updates.

The caller owns the batch stream, metric resets, callbacks, epoch boundaries,
random state and thread limits. This does not change the model's train_step or
test_step implementation, optimizer, stochastic layers or batch geometry.
"""
    import tensorflow as tf
    import tf_keras
    if not isinstance(model, tf_keras.Model) or model.optimizer is None:
        raise ValueError('A compiled legacy-Keras model is required')

    @tf.function(reduce_retracing=True)
    def train(inputs, targets):
        return model.train_step((inputs, targets))

    @tf.function(reduce_retracing=True)
    def validate(inputs, targets):
        return model.test_step((inputs, targets))

    return train, validate

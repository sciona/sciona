"""OSIC source quantile architecture with an explicit legacy-Keras optimizer.

Feature preparation, availability and column meaning belong to the caller. The
three outputs are not guaranteed to be ordered. No fitting happens at creation.
"""
from sciona.osic_quantile_objective import mixed_loss, optimizer, score


def build_model(feature_count):
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count < 1:
        raise ValueError('Positive integer feature count required')
    import tensorflow as tf
    import tf_keras as keras
    inputs = keras.layers.Input((feature_count,), name='features')
    hidden = keras.layers.Dense(100, activation='relu', name='d1')(inputs)
    hidden = keras.layers.Dense(100, activation='relu', name='d2')(hidden)
    base = keras.layers.Dense(3, activation='linear', name='p1')(hidden)
    increments = keras.layers.Dense(3, activation='relu', name='p2')(hidden)
    predictions = keras.layers.Lambda(lambda values: values[0] + tf.cumsum(values[1], axis=1),
                                     name='preds')([base, increments])
    model = keras.Model(inputs, predictions, name='osic_quantile')
    model.compile(loss=mixed_loss, optimizer=optimizer(), metrics=[score])
    return model

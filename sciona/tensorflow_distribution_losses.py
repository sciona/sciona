"""Differentiable distribution objectives with caller-owned coordinates/policies.

These functions preserve float32 TensorFlow arithmetic and gradients. They are
not registered catalog atoms yet. Targets are nonempty rows-by-one tensors;
prediction columns and target units must be aligned by the caller.
"""


def _inputs(targets, predictions):
    import tensorflow as tf
    targets = tf.convert_to_tensor(targets)
    predictions = tf.convert_to_tensor(predictions)
    if targets.dtype != tf.float32 or predictions.dtype != tf.float32:
        raise ValueError('Explicit float32 target and prediction tensors required')
    tf.debugging.assert_rank(targets, 2)
    tf.debugging.assert_rank(predictions, 2)
    tf.debugging.assert_equal(tf.shape(targets)[1], 1)
    tf.debugging.assert_equal(tf.shape(targets)[0], tf.shape(predictions)[0])
    tf.debugging.assert_positive(tf.shape(targets)[0])
    tf.debugging.assert_all_finite(targets, 'Finite targets required')
    tf.debugging.assert_all_finite(predictions, 'Finite predictions required')
    return targets, predictions


def mean_pinball(targets, predictions, quantiles):
    """Mean pinball loss over rows and explicit ordered prediction columns.

Quantiles must be finite and strictly between zero and one. The objective does
not enforce ordering of predictions or establish calibrated interval coverage.
"""
    import tensorflow as tf
    targets, predictions = _inputs(targets, predictions)
    q = tf.convert_to_tensor(quantiles, dtype=tf.float32)
    tf.debugging.assert_rank(q, 1)
    tf.debugging.assert_equal(tf.shape(q)[0], tf.shape(predictions)[1])
    tf.debugging.assert_positive(tf.shape(q)[0])
    tf.debugging.assert_greater(q, 0.)
    tf.debugging.assert_less(q, 1.)
    residual = targets - predictions
    return tf.reduce_mean(tf.maximum(q[None, :] * residual, (q[None, :] - 1) * residual))


def clipped_laplace_score(targets, predictions, minimum_scale, maximum_error):
    """Mean clipped Laplace-style score for low/point/high prediction columns.

Scale is high minus low, floored by the positive caller-supplied minimum.
Absolute point error is capped by a positive maximum in the same target units.
Negative/crossed raw spreads are accepted and encounter the scale floor. This
scoring policy alone is not a probabilistic calibration guarantee.
"""
    import math
    import tensorflow as tf
    if (not isinstance(minimum_scale, (float, int)) or isinstance(minimum_scale, bool)
            or not isinstance(maximum_error, (float, int)) or isinstance(maximum_error, bool)
            or not math.isfinite(minimum_scale) or not math.isfinite(maximum_error)
            or minimum_scale <= 0 or maximum_error <= 0):
        raise ValueError('Positive finite scale floor and error cap required')
    targets, predictions = _inputs(targets, predictions)
    tf.debugging.assert_equal(tf.shape(predictions)[1], 3)
    floor = tf.constant(minimum_scale, dtype=tf.float32)
    cap = tf.constant(maximum_error, dtype=tf.float32)
    for value in (floor, cap):
        tf.debugging.assert_all_finite(value, 'Float32 policy constants required')
        tf.debugging.assert_positive(value)
    scale = predictions[:, 2] - predictions[:, 0]
    scale = tf.maximum(scale, floor)
    error = tf.minimum(tf.abs(targets[:, 0] - predictions[:, 1]),
                       cap)
    root_two = tf.sqrt(tf.constant(2, dtype=tf.float32))
    return tf.reduce_mean(error / scale * root_two + tf.math.log(scale * root_two))

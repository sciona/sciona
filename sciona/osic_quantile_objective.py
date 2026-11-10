"""OSIC source constants kept separate from general differentiable losses."""
from sciona.tensorflow_distribution_losses import mean_pinball, clipped_laplace_score


def quantile_loss(targets, predictions):
    return mean_pinball(targets, predictions, [.2, .5, .8])


def score(targets, predictions):
    return clipped_laplace_score(targets, predictions, 70., 1000.)


def mixed_loss(targets, predictions):
    # Preserve the source expression 1 - 0.8 before conversion to TensorFlow.
    return .8 * quantile_loss(targets, predictions) + (1 - .8) * score(targets, predictions)


def optimizer():
    """Explicit legacy-Keras compatibility choice; no modern Adam substitution.

Dense float32 update equivalence must be qualified separately. This creates
state only; it does not initialize or train either competition model.
"""
    import tf_keras
    return tf_keras.optimizers.legacy.Adam(learning_rate=.1, beta_1=.9, beta_2=.999,
        epsilon=None, decay=.01, amsgrad=False)

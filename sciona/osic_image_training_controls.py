"""Source-specific OSIC image training policy, separate from model topology."""
from pathlib import Path

EPOCHS = 30
BATCH_SIZE = 8
TRAINING_STEPS = 32
VALIDATION_STEPS = 16


def compile_model(model):
    """Compile for the explicitly selected legacy Adam/float32 runtime."""
    import tf_keras
    if not isinstance(model, tf_keras.Model):
        raise ValueError('A legacy-Keras model is required')
    model.compile(optimizer=tf_keras.optimizers.legacy.Adam(
        learning_rate=.003, beta_1=.9, beta_2=.999, epsilon=None,
        decay=0., amsgrad=False), loss='mae')


def callbacks(checkpoint):
    """Save a private full-model HDF5 checkpoint before updating the LR.

The source constructs but does not activate early stopping. The checkpoint
contains optimizer state as it stood before that epoch's LR callback. The
caller owns the destination and must not publish fitted state.
"""
    import tf_keras
    checkpoint = Path(checkpoint)
    if checkpoint.suffix != '.h5':
        raise ValueError('The qualified full-model HDF5 checkpoint format is required')
    return [tf_keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor='val_loss', verbose=0,
                save_best_only=True, mode='auto'),
            tf_keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=.5, patience=5, verbose=0,
                min_lr=1e-8)]

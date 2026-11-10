"""Complete source image schedule over explicitly prepared populations."""
from pathlib import Path

import numpy as np

from sciona.keras_serial_steps import make_steps
from sciona.osic_image_batches import validate_population, sample_batch
from sciona.osic_image_model import build_model
from sciona import osic_image_training_controls as controls


def train(images, metadata, slopes, *, seed, checkpoint_directory, progress=None):
    """Run 30 epochs, 32 training and 16 validation batches of eight per epoch.

Source training and validation draw from the same population. Sequential
sampling has an explicit seed; it does not reconstruct historical prefetch
order or the random stream consumed by unused backbone constructions. Raw
decoding and domain preparation are upstream. Checkpoints stay private.
"""
    import tf_keras
    validate_population(images, metadata, slopes)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('An explicit uint32-compatible integer seed is required')
    directory = Path(checkpoint_directory)
    directory.mkdir(parents=True, exist_ok=False)
    tf_keras.utils.set_random_seed(seed)
    random_state = np.random.RandomState(seed)
    model = build_model()
    controls.compile_model(model)
    training_step, validation_step = make_steps(model)
    selected = directory/'best.h5'
    callbacks = controls.callbacks(selected)
    for callback in callbacks:
        callback.set_model(model)
        callback.set_params(dict(epochs=controls.EPOCHS,steps=controls.TRAINING_STEPS,verbose=0))
        callback.on_train_begin({})
    history = []
    for epoch in range(controls.EPOCHS):
        for callback in callbacks:
            callback.on_epoch_begin(epoch,{})
        model.reset_metrics()
        for batch in range(controls.TRAINING_STEPS):
            inputs, targets = sample_batch(images,metadata,slopes,random_state)
            logs = training_step(inputs,targets)
            if not np.isfinite(float(logs['loss'])):
                raise FloatingPointError('Nonfinite image training loss; no qualification')
        training_loss = float(logs['loss'])
        model.reset_metrics()
        for batch in range(controls.VALIDATION_STEPS):
            inputs, targets = sample_batch(images,metadata,slopes,random_state)
            logs = validation_step(inputs,targets)
            if not np.isfinite(float(logs['loss'])):
                raise FloatingPointError('Nonfinite image validation loss; no qualification')
        epoch_logs = dict(loss=training_loss,val_loss=float(logs['loss']))
        for callback in callbacks:
            callback.on_epoch_end(epoch,epoch_logs)
        entry = dict(epoch=epoch+1,loss=training_loss,val_loss=epoch_logs['val_loss'],
            learning_rate=float(tf_keras.backend.get_value(model.optimizer.learning_rate)),
            optimizer_iterations=int(model.optimizer.iterations))
        history.append(entry)
        if progress is not None:
            progress(entry)
    for callback in callbacks:
        callback.on_train_end({})
    if int(model.optimizer.iterations) != 960 or not selected.is_file():
        raise ValueError('Incomplete source training schedule')
    return model, history, selected

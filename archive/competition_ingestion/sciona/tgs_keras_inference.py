"""Source-ordered Keras TGS TTA and per-checkpoint uint8 quantization.

Independent array realization of pinned bes/utils.py. No temporary prediction
images are written; uint8 conversion preserves their numerical boundary.
"""
import cv2
import numpy as np
import torch

from sciona.tgs_keras_preprocessing import prepare_keras


def predict_checkpoints(model, states, images, *, batch_size):
    """Return checkpoint/population/101/101 probabilities after source quantization.

    Sigmoid-mode prediction uses the same learned head parameters as logit-mode
    training. Fold and snapshot blending occurs after per-checkpoint quantization.
    Model is left in probability/eval mode with its last supplied state.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('positive integer batch size required')
    if not isinstance(states, (list, tuple)) or not states:
        raise ValueError('nonempty ordered checkpoint states required')
    values = np.asarray(images)
    if not len(values):
        raise ValueError('nonempty query population required')
    for start in range(0, len(values), batch_size):
        prepare_keras(values[start:start+batch_size])
    result = []
    model.probabilities = True
    model.eval()
    for state in states:
        model.load_state_dict(state, strict=True)
        checkpoint = []
        for start in range(0, len(values), batch_size):
            batch = values[start:start+batch_size]
            interleaved = np.stack([view for image in batch for view in (image, image[:, ::-1])])
            prepared = prepare_keras(interleaved)
            with torch.no_grad():
                predicted = model(prepared['images'])
            if predicted.shape != (2 * len(batch), 1, 224, 224) or not torch.isfinite(predicted).all() or (predicted < 0).any() or (predicted > 1).any():
                raise ValueError('aligned finite model probabilities required')
            cropped = predicted[:, 0, 16:208, 16:208].detach().cpu().numpy()
            for index in range(len(batch)):
                plain = cv2.resize(cropped[2 * index], (101, 101), interpolation=cv2.INTER_LINEAR)
                flipped = cv2.resize(cropped[2 * index + 1], (101, 101), interpolation=cv2.INTER_LINEAR)[:, ::-1]
                averaged = np.mean(np.stack((plain, flipped)), axis=0)
                checkpoint.append((averaged * 255).astype(np.uint8).astype(np.float32) / 255.)
        result.append(np.stack(checkpoint))
    return np.stack(result)

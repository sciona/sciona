"""Checkpoint/TTA inference for the corrected TGS PyTorch branch.

Matches the operation ordering in pinned phalanx/precisioncv.py. Input states
are internally reviewed model state dictionaries, not external serialized data.
"""
import cv2
import numpy as np
import torch

from sciona.tgs_preprocessing import prepare_torch


def predict_snapshots(model, states, images, *, batch_size):
    """Average sigmoid probabilities over horizontal TTA and snapshots.

    Leaves model in eval mode with the last supplied state loaded. Fold averaging
    is a later stage. Variant5 flip occurs before its asymmetric input padding;
    only the cropped predictions are flipped back.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('positive integer inference batch size required')
    if not isinstance(states, (tuple, list)) or not states:
        raise ValueError('nonempty ordered checkpoint state sequence required')
    values = np.asarray(images)
    if not len(values):
        raise ValueError('nonempty inference population required')
    # Validate complete input before loading any state.
    for start in range(0, len(values), batch_size):
        prepare_torch(values[start:start+batch_size], variant=model.variant)
    size = 101 if model.variant == 5 else 202
    average = np.zeros((len(values), size, size), dtype=np.float64)
    model.eval()
    for state in states:
        model.load_state_dict(state, strict=True)
        views = []
        for flip in (False, True):
            view = np.empty((len(values), size, size), dtype=np.float32)
            for start in range(0, len(values), batch_size):
                prepared = prepare_torch(values[start:start+batch_size], variant=model.variant, flip=flip)
                with torch.no_grad():
                    output = model(prepared['images'])
                    if model.variant == 3:
                        output = output[0]
                    if not torch.isfinite(output).all():
                        raise ValueError('nonfinite inference logits')
                    probability = output.sigmoid()[:, 0].cpu().numpy()
                offset = prepared['crop_start']
                cropped = probability[:, offset:offset+size, offset:offset+size]
                if cropped.shape != (len(prepared['images']), size, size):
                    raise ValueError('unexpected inference output shape')
                if flip:
                    cropped = cropped[:, :, ::-1]
                view[start:start+len(cropped)] = cropped
            views.append(view)
        average += (views[0] + views[1]) / 2
    average /= len(states)
    if size != 101:
        average = np.stack([cv2.resize(item, (101, 101), interpolation=cv2.INTER_LINEAR) for item in average])
    return average

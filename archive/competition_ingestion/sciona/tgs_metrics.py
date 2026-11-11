"""Explicit TGS Keras validation metric and uncapped diagnostic.

Source bes/losses.py at pinned winner revision clips union to 128 squared.
Preserve that policy for source-comparable callback decisions; also expose the
uncapped diagnostic to reveal distortion at larger input sizes.
"""
import numpy as np
import torch
from sciona.tgs_losses import keras_mask_pair
from sciona.hubmap_losses import _mask_pair


def torch_validation_metric(logits, targets):
    """Source binary-mask metric, including denominator epsilon and empty cases."""
    _mask_pair(logits, targets)
    prediction = logits.detach().sigmoid().numpy().reshape(len(logits), -1) > .5
    truth = targets.detach().numpy().reshape(len(targets), -1) > .5
    intersection = (prediction & truth).sum(1)
    union = (prediction | truth).sum(1)
    iou = intersection / (union + 1e-8)
    thresholds = np.array([.50, .55, .60, .65, .70, .75, .80, .85, .90, .95])
    per_image = np.where(union == 0, 1., (iou[:, None] > thresholds).mean(1))
    return dict(mean=float(per_image.mean()), per_image=per_image)


def keras_validation_metric(predictions, targets, *, score_kind, source_union_cap):
    keras_mask_pair(predictions, targets)
    if score_kind not in ('logits', 'probabilities') or not isinstance(source_union_cap, bool):
        raise ValueError('explicit score kind and union policy required')
    if score_kind == 'probabilities' and ((predictions < 0) | (predictions > 1)).any():
        raise ValueError('probabilities must be in [0,1]')
    with torch.no_grad():
        predicted = (predictions > (0 if score_kind == 'logits' else .5)).float().flatten(1)
        truth = targets.flatten(1)
        total_area = truth.sum(1) + predicted.sum(1)
        intersection = (truth * predicted).sum(1)
        union = (total_area - intersection).clamp(min=1e-9)
        if source_union_cap:
            union = union.clamp(max=128 ** 2)
        iou = torch.where(total_area == 0, torch.ones_like(union), intersection / union)
        thresholds = torch.tensor(np.arange(.5, 1., .05), dtype=torch.float32)
        per_image = (iou[:, None] > thresholds).float().mean(1)
        return dict(mean=float(per_image.mean()), per_image=per_image, iou=iou)

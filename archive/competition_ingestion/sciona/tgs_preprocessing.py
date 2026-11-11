"""PyTorch-branch TGS resize, edge-padding and position-channel contract.

Source transform and dataset files are pinned in the TGS source review.
Training augmentation is a separate required preceding stage.
"""
import cv2
import numpy as np
import torch


def prepare_torch(images, masks=None, *, variant, training=False, flip=False):
    if isinstance(variant, bool) or variant not in (3, 4, 5):
        raise ValueError('variant must be 3, 4 or 5')
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 3 or not len(values) or values.shape[1:] != (101, 101):
        raise ValueError('nonempty N101x101 grayscale input required')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('normalized finite grayscale input required')
    labels = None
    if masks is not None:
        labels = np.asarray(masks, dtype=np.float32)
        if labels.shape != values.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError('aligned binary masks required')
    if training and labels is None:
        raise ValueError('training requires masks')
    if flip and labels is not None:
        raise ValueError('inference flip is only valid without masks')
    size, padding = (101, (13, 14)) if variant == 5 else (202, (27, 27))
    prepared, targets = [], []
    for i, image in enumerate(values):
        if flip:
            image = image[:, ::-1]
        if size != 101:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        image = np.pad(image, padding, mode='edge')
        vertical = np.broadcast_to(np.linspace(0, 1, len(image), dtype=np.float32)[:, None], image.shape)
        prepared.append(np.stack((image, vertical, image * vertical)))
        if labels is not None:
            mask = labels[i]
            if size != 101:
                mask = (cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR) > .5).astype(np.float32)
            if training:
                mask = np.pad(mask, padding, mode='edge')
            targets.append(mask[None])
    result = dict(images=torch.from_numpy(np.stack(prepared)), crop_start=padding[0], crop_size=size)
    if labels is not None:
        result['masks'] = torch.from_numpy(np.stack(targets))
        result['empty_image_targets'] = torch.from_numpy((labels.sum(axis=(1, 2)) == 0).astype(np.float32))
    return result

"""Uncropped classifier input adapter with explicitly supplied normalization.

The fixed geometry belongs to the reconstructed grapheme classifier. Reusing
this adapter requires compatible grayscale inputs; it is not a universal image
preprocessor. Mean/std are caller policy, not recovered winner settings.
"""
import numpy as np
from PIL import Image
import torch

from sciona.svhn_autoaugment import SVHNAutoAugment


def prepare_classifier_images(images, *, mean, std, augmentation):
    """Replicate grayscale channels, optionally augment, then normalize RGB.

    Supply augmentation=None explicitly for deterministic evaluation, or a
    persistent SVHNAutoAugment instance for a caller-owned training RNG stream.
    Input validation completes before that stream advances.
    """
    images = np.asarray(images)
    if (images.dtype != np.uint8 or images.ndim != 3 or not len(images)
            or images.shape[1:] != (137, 236)):
        raise ValueError('Nonempty uint8 grayscale N137x236 images required')
    channels = []
    for value in (mean, std):
        array = np.asarray(value)
        if array.shape != (3,) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError('Explicit finite three-channel mean/std required')
        with np.errstate(over='ignore'):
            array = array.astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError('Normalization must be representable as float32')
        channels.append(array)
    mean, std = channels
    if (std <= 0).any():
        raise ValueError('Positive channel standard deviations required')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        bounds = (np.array([0., 1.], dtype=np.float32)[:, None] - mean) / std
    if not np.isfinite(bounds).all():
        raise ValueError('Normalization must remain finite over the byte range')
    if augmentation is not None and not isinstance(augmentation, SVHNAutoAugment):
        raise ValueError('Explicit SVHN augmentation instance or None required')
    rgb = np.repeat(images[..., None], 3, axis=3)
    if augmentation is not None:
        rgb = np.stack([np.asarray(augmentation(Image.fromarray(image))) for image in rgb])
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))).float()
    tensor.div_(255.).sub_(torch.from_numpy(mean)[None, :, None, None])
    tensor.div_(torch.from_numpy(std)[None, :, None, None])
    return tensor

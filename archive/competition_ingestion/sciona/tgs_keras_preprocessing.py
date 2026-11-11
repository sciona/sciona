"""Independent TGS Keras array preprocessing before/after augmentation.

Source generator and preprocessing dispatch are pinned in the source review.
Input image values retain the source BGR 0..255 convention, with no mean/std
normalization. Augmentation belongs before this stage, as in the source.
"""
import cv2
import numpy as np
import torch


def prepare_keras(images, masks=None):
    """Resize 101->192, reflect101-pad 16 per side, return NCHW tensors.

    Images are N101x101x3 BGR uint8 or floating arrays in [0,255]. Masks
    deliberately remain uint8 through interpolation, preserving source rounding
    before conversion to fractional coverage. No thresholding is performed.
    """
    image = np.asarray(images)
    if image.ndim != 4 or not len(image) or image.shape[1:] != (101, 101, 3):
        raise ValueError('nonempty N101x101x3 BGR images required')
    if not np.isfinite(image).all() or (image < 0).any() or (image > 255).any():
        raise ValueError('finite image values in [0,255] required')
    mask = None if masks is None else np.asarray(masks)
    if mask is not None and (mask.dtype != np.uint8 or mask.shape != image.shape[:3]):
        raise ValueError('aligned uint8 coverage masks required')
    prepared, coverage = [], []
    for index, sample in enumerate(image):
        resized = cv2.resize(sample.astype(np.float32), (192, 192), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
        prepared.append(padded.transpose(2, 0, 1))
        if mask is not None:
            resized_mask = cv2.resize(mask[index], (192, 192), interpolation=cv2.INTER_LINEAR)
            padded_mask = cv2.copyMakeBorder(resized_mask, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
            coverage.append(padded_mask[None].astype(np.float32) / 255.)
    result = dict(images=torch.from_numpy(np.stack(prepared)), crop_start=16, crop_size=192)
    if coverage:
        result['masks'] = torch.from_numpy(np.stack(coverage))
    return result

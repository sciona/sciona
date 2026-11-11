"""Source-compatible Albumentations 0.1.2 TGS recipe, independently expressed.

Preserves the historical float32 clipping to [0,1] in brightness/contrast even
though the winner passes 0..255 float32 images. This is observable source
behavior, not a recommended general-purpose image augmentation convention.
"""
import cv2
import numpy as np


def photometric(image, *, brightness=None, contrast=None):
    value = np.asarray(image, dtype=np.float32).copy()
    if value.shape != (101, 101, 3) or not np.isfinite(value).all() or (value < 0).any() or (value > 255).any():
        raise ValueError('finite 101-square BGR image in [0,255] required')
    for amount in (brightness, contrast):
        if amount is not None and (not np.isfinite(amount) or not .8 <= amount <= 1.2):
            raise ValueError('source photometric factor must be in [.8,1.2]')
    if brightness is not None:
        value = np.clip(value * brightness, 0, 1)
    if contrast is not None:
        # Source RGB conversion is preserved despite caller BGR channel order.
        mean = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY).mean()
        value = np.clip(contrast * value + 3 * (1 - contrast) * mean, 0, 1)
    return value.astype(np.float32)


def augment_keras(image, mask, rng):
    if not isinstance(rng, np.random.Generator):
        raise ValueError('explicit NumPy Generator required')
    value = photometric(image)
    target = np.asarray(mask)
    if target.shape != (101, 101) or target.dtype != np.uint8:
        raise ValueError('aligned uint8 mask required')
    target = target.copy()
    if rng.random() < .5:
        value, target = value[:, ::-1].copy(), target[:, ::-1].copy()
    brightness = float(rng.uniform(.8, 1.2)) if rng.random() < .2 else None
    contrast = float(rng.uniform(.8, 1.2)) if rng.random() < .1 else None
    value = photometric(value, brightness=brightness, contrast=contrast)
    if rng.random() < .7:
        scale = float(rng.uniform(.4, 1.6))
        dx, dy = rng.uniform(-.1625, .1625, 2)
        matrix = cv2.getRotationMatrix2D((50.5, 50.5), 0, scale)
        matrix[:, 2] += np.array([dx, dy]) * 101
        value = cv2.warpAffine(value, matrix, (101, 101), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        target = cv2.warpAffine(target, matrix, (101, 101), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return value, target

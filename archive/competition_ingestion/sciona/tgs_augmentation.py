"""Independent PyTorch-branch TGS augmentation recipe on synthetic-safe arrays.

Method source pinned in TGS review. Explicit NumPy Generator replaces global
random state; historical random-sequence equivalence is not claimed.
"""
import math
import cv2
import numpy as np


def transform_pair(image, mask, *, flip=False, geometry=None, parameter=None,
                   brightness=None, amount=0.):
    value, target = np.asarray(image, dtype=np.float32), np.asarray(mask, dtype=np.float32)
    if value.shape != (101, 101) or target.shape != value.shape:
        raise ValueError('aligned 101-square arrays required')
    if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any() or not np.isin(target, [0, 1]).all():
        raise ValueError('normalized image and binary mask required')
    value, target = value.copy(), target.copy()
    if flip:
        value, target = value[:, ::-1].copy(), target[:, ::-1].copy()
    if geometry == 'crop':
        y0, y1, x0, x1 = parameter
        if not all(isinstance(v, (int, np.integer)) for v in parameter) or not 0 <= y0 < y1 <= 101 or not 0 <= x0 < x1 <= 101:
            raise ValueError('valid integer crop rectangle required')
        value = cv2.resize(value[y0:y1, x0:x1], (101, 101), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target[y0:y1, x0:x1], (101, 101), interpolation=cv2.INTER_LINEAR)
    elif geometry in ('shear', 'rotate'):
        if not math.isfinite(parameter):
            raise ValueError('finite geometric parameter required')
        corners = np.array([[0, 0], [101, 0], [101, 101], [0, 101]], dtype=np.float32)
        if geometry == 'shear':
            displacement = int(parameter * 101)
            destination = corners + np.array([[displacement, 0], [displacement, 0], [-displacement, 0], [-displacement, 0]])
        else:
            angle = math.radians(parameter)
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            destination = (corners - 50.5) @ rotation.T + 50.5
        matrix = cv2.getPerspectiveTransform(corners, destination.astype(np.float32))
        value = cv2.warpPerspective(value, matrix, (101, 101), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        target = cv2.warpPerspective(target, matrix, (101, 101), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    elif geometry is not None:
        raise ValueError('unknown geometry')
    target = (target > .5).astype(np.float32)
    if not math.isfinite(amount):
        raise ValueError('finite brightness parameter required')
    if brightness == 'shift':
        value = np.clip(value + amount, 0, 1)
    elif brightness == 'multiply':
        value = np.clip(value * amount, 0, 1)
    elif brightness is not None:
        raise ValueError('unknown brightness transform')
    return value, target


def augment_torch(image, mask, rng):
    """Draw the source transform probabilities/ranges from explicit RNG."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError('explicit NumPy Generator required')
    options = dict(flip=bool(rng.random() < .5))
    if rng.random() < .5:
        choice = int(rng.integers(3))
        if choice == 0:
            cuts = rng.integers(0, 20, size=4)
            options.update(geometry='crop', parameter=(int(cuts[0]), 101-int(cuts[1]), int(cuts[2]), 101-int(cuts[3])))
        elif choice == 1:
            options.update(geometry='shear', parameter=float(rng.uniform(-.07, .07)))
        else:
            options.update(geometry='rotate', parameter=float(rng.uniform(0, 15)))
    if rng.random() < .5:
        if rng.integers(2) == 0:
            options.update(brightness='shift', amount=float(rng.uniform(-.1, .1)))
        else:
            options.update(brightness='multiply', amount=float(rng.uniform(.92, 1.08)))
    return transform_pair(image, mask, **options)

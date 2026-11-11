"""Winner-ordered image permutations and inverse xyxy transformations.

Adapted from DungNB (Nguyen Ba Dung),2020, under the MIT license retained
in docs/licenses/global-wheat-MIT.txt. Source horizontal/vertical names refer
to height/width flips respectively; their actual tensor axes are preserved.
"""
from itertools import product

import numpy as np
import torch


VIEWS = tuple(product((True, False), repeat=3))


class WheatTTA:
    def __init__(self, *, view, image_size):
        if type(view) is not int or not 0 <= view < 8 or type(image_size) is not int or image_size < 1:
            raise ValueError('view0through7 and positive square image size required')
        self.view, self.image_size = view, image_size

    def augment_tensor(self, images):
        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1
                or images.shape[1:] != (3, self.image_size, self.image_size)
                or not torch.isfinite(images).all()):
            raise ValueError('finite nonempty N3SS image tensor required')
        horizontal, vertical, rotate = VIEWS[self.view]
        if horizontal:
            images = images.flip(2)
        if vertical:
            images = images.flip(3)
        if rotate:
            images = torch.rot90(images, 1, (2, 3))
        return images

    def augment_fasterrcnn(self, images):
        return list(self.augment_tensor(torch.stack(images)).unbind(0))

    @staticmethod
    def _ordered(boxes):
        result = boxes.copy()
        result[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result

    def deaugment_boxes(self, boxes):
        boxes = np.asarray(boxes)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,) or boxes.dtype.kind != 'f' or not np.isfinite(boxes).all():
            raise ValueError('finite floating-point Nx4 boxes required')
        boxes = boxes.copy()
        horizontal, vertical, rotate = VIEWS[self.view]
        if rotate:
            result = boxes.copy()
            result[:, [0, 2]] = self.image_size - boxes[:, [1, 3]]
            result[:, [1, 3]] = boxes[:, [2, 0]]
            boxes = self._ordered(result)
        if vertical:
            boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        if horizontal:
            boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
            boxes = self._ordered(boxes)
        return self._ordered(boxes)

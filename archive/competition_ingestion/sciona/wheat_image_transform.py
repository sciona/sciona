"""Historical box-detector resize and target mutation semantics."""
import math

import torch
from torch.nn import functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def resize_boxes(boxes, original_size, new_size):
    ratio_h, ratio_w = [float(new) / float(old) for new, old in zip(new_size, original_size)]
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack([x1 * ratio_w, y1 * ratio_h, x2 * ratio_w, y2 * ratio_h], dim=1)


class WheatImageTransform(GeneralizedRCNNTransform):
    def __init__(self):
        super().__init__(800, 1333, [.485, .456, .406], [.229, .224, .225])

    def resize(self, image, target=None):
        original = image.shape[-2:]
        size = float(self.torch_choice(self.min_size) if self.training else self.min_size[-1])
        scale = size / min(original)
        if max(original) * scale > self.max_size:
            scale = self.max_size / max(original)
        shape = [int(math.floor(float(n) * scale)) for n in original]
        image = F.interpolate(image[None], size=shape, mode='bilinear', align_corners=False)[0]
        if target is not None:
            if 'masks' in target or 'keypoints' in target:
                raise ValueError('Global Wheat uses bounding-box targets only')
            target['boxes'] = resize_boxes(target['boxes'], original, image.shape[-2:])
        return image, target

    def forward(self, images, targets=None):
        resized = []
        for index, image in enumerate(images):
            if image.ndim != 3:
                raise ValueError('three-dimensional image tensors required')
            target = targets[index] if targets is not None else None
            image, target = self.resize(self.normalize(image), target)
            resized.append(image)
            if targets is not None:
                targets[index] = target
        sizes = [tuple(image.shape[-2:]) for image in resized]
        return ImageList(self.batch_images(resized), sizes), targets

    def postprocess(self, result, image_shapes, original_image_sizes):
        if not self.training:
            for prediction, resized, original in zip(result, image_shapes, original_image_sizes):
                prediction['boxes'] = resize_boxes(prediction['boxes'], resized, original)
        return result

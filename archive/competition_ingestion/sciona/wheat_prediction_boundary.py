"""Source detector-specific score/box conversion before Global Wheat fusion.

Adapted from DungNB (Nguyen Ba Dung),2020; MIT notice is retained in
docs/licenses/global-wheat-MIT.txt. Detector execution remains caller-owned.
"""
import numpy as np

from sciona.wheat_tta import WheatTTA


def prepare_prediction(boxes, scores, *, detector, view, image_size):
    if detector not in ('effdet', 'fasterrcnn'):
        raise ValueError('source detector kind required')
    transform = WheatTTA(view=view, image_size=image_size)
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    if (boxes.ndim != 2 or boxes.shape[1:] != (4,) or scores.shape != (len(boxes),)
            or boxes.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or scores.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise ValueError('finite floating detector boxes and aligned probabilities required')
    boxes, scores = boxes.copy(), scores.copy()
    if detector == 'fasterrcnn':
        scores *= .8
    else:
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    boxes = transform.deaugment_boxes(boxes)
    keep = np.where(scores > .2)[0]
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        boxes /= float(image_size)
        boxes = boxes.clip(min=0, max=1)
    return boxes, scores, np.zeros_like(scores)

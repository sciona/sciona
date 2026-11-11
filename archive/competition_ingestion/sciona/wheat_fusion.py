"""Global Wheat's source-stage WBF thresholds and pixel-coordinate boundary.

Inputs are the ordered, normalized predictions after detector-specific score
adjustment and TTA reversal. Full prediction generation is a separate stage.
"""
from importlib.metadata import version

import numpy as np


STAGES = {'initial': (72, .32, .28), 'pseudo1': (8, .42, .32), 'pseudo2': (8, .44, .34)}


def fuse_predictions(boxes_by_view, scores_by_view, *, stage, height, width):
    if stage not in STAGES:
        raise ValueError('explicit initial, pseudo1 or pseudo2 fusion stage required')
    if any(type(v) is not int or v < 1 for v in (height, width)):
        raise ValueError('positive original pixel dimensions required')
    count, box_threshold, post_threshold = STAGES[stage]
    if len(boxes_by_view) != count or len(scores_by_view) != count:
        raise ValueError('complete ordered detector/TTA view inventory required')
    boxes, scores, labels = [], [], []
    for raw_boxes, raw_scores in zip(boxes_by_view, scores_by_view):
        b, s = np.asarray(raw_boxes), np.asarray(raw_scores)
        if b.shape == (0,):
            b = np.empty((0, 4))
        if (b.ndim != 2 or b.shape[1:] != (4,) or s.shape != (len(b),)
                or b.dtype.kind not in 'fiu' or s.dtype.kind not in 'fiu'
                or not np.isfinite(b).all() or not np.isfinite(s).all()
                or (b < 0).any() or (b > 1).any() or (b[:, 2:] < b[:, :2]).any()
                or (s < 0).any() or (s > 1).any()):
            raise ValueError('finite aligned normalized xyxy boxes and confidence required')
        boxes.append(b.tolist())
        scores.append(s.tolist())
        labels.append([0] * len(b))
    if version('ensemble-boxes') != '1.0.4':
        raise ValueError('qualified ensemble-boxes1.0.4 required')
    from ensemble_boxes import weighted_boxes_fusion
    # Empty views must remain in these lists: they affect source confidence.
    b, s, _ = weighted_boxes_fusion(boxes, scores, labels, weights=None,
                                    iou_thr=.5, skip_box_thr=box_threshold)
    keep = np.where(s > post_threshold)[0]
    b, s = np.array(b)[keep], np.array(s)[keep]
    if len(b):
        b[:, [0, 2]] = (b[:, [0, 2]] * width).clip(min=0, max=width - 1)
        b[:, [1, 3]] = (b[:, [1, 3]] * height).clip(min=0, max=height - 1)
    return b, s

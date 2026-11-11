"""Source validation score for the Global Wheat Faster R-CNN base fit."""
import numpy as np


def _image_precision(ground_truth, predictions):
    value = 0.0
    for threshold in (.5, .55, .6, .65, .7, .75):
        remaining = ground_truth.copy()
        true_positive = false_positive = 0
        for prediction in predictions:
            best_index, best_iou = -1, -np.inf
            for index, target in enumerate(remaining):
                if target[0] < 0:
                    continue
                dx = min(target[2], prediction[2]) - max(target[0], prediction[0]) + 1
                dy = min(target[3], prediction[3]) - max(target[1], prediction[1]) + 1
                if dx < 0 or dy < 0:
                    iou = 0.0
                else:
                    overlap = dx * dy
                    union = ((target[2]-target[0]+1)*(target[3]-target[1]+1)
                             + (prediction[2]-prediction[0]+1)*(prediction[3]-prediction[1]+1) - overlap)
                    iou = overlap / union
                if iou >= threshold and iou > best_iou:
                    best_index, best_iou = index, iou
            if best_index >= 0:
                true_positive += 1
                remaining[best_index] = -1
            else:
                false_positive += 1
        # Preserve the source's sum-based remaining-box count, including origin boxes.
        false_negative = (remaining.sum(axis=1) > 0).sum()
        denominator = true_positive + false_positive + false_negative
        precision = 1.0 if denominator == 0 else true_positive / denominator
        value += precision / 6
    return value


def fasterrcnn_validation_score(records, image_size=1024):
    """Records contain raw predicted xyxy boxes, scores and ground-truth xyxy boxes.

    Clip and integer conversion precede matching, as in the training script.
    Preserve prediction order; the source does not sort by score here.
    """
    if not records or type(image_size) is not int or image_size < 1:
        raise ValueError('nonempty validation records and positive image size required')
    values = []
    for record in records:
        boxes, scores, truth = [np.asarray(record[key]) for key in ('pred_boxes', 'scores', 'gt_boxes')]
        if (boxes.ndim != 2 or boxes.shape[1:] != (4,) or truth.ndim != 2 or truth.shape[1:] != (4,)
                or scores.shape != (len(boxes),) or any(not np.isfinite(v).all() for v in (boxes, scores, truth))):
            raise ValueError('finite aligned box and score arrays required')
        selected = boxes.clip(0, image_size-1).astype(int)[scores > .5]
        ground_truth = truth.astype(int)
        if (ground_truth < 0).any() or (ground_truth[:, 2:] < ground_truth[:, :2]).any() or (selected[:, 2:] < selected[:, :2]).any():
            raise ValueError('ordered nonnegative source box coordinates required')
        values.append(_image_precision(ground_truth, selected))
    return float(np.mean(values))

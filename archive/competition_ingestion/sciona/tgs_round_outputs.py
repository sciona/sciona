"""TGS ensemble-to-pseudo-label barriers and final mosaic output boundary."""
import numpy as np

from sciona.tgs_postprocessing import postprocess
from sciona.tgs_pseudo_selection import select_pseudo_labels


def complete_round(stage, scores, query_images, *, training_images=None, training_masks=None):
    """Derive masks/statistics from the full ordered query ensemble population.

    Earlier rounds emit binary pseudo-masks for every query position, plus
    distinct branch selections. Only the final round propagates mosaic masks.
    No records, source identifiers, images or checkpoint files are written.
    """
    if type(stage) is not int or stage not in (1, 2, 3):
        raise ValueError('stage must be 1, 2 or 3')
    scores, query = np.asarray(scores), np.asarray(query_images)
    if (query.ndim != 3 or query.shape[0] < 1 or query.shape[1:] != (101, 101)
            or scores.shape != query.shape or not np.isfinite(query).all()
            or (query < 0).any() or (query > 1).any()
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
        raise ValueError('aligned finite normalized N101x101 query images and scores required')
    if stage == 3:
        if training_images is None or training_masks is None:
            raise ValueError('final round requires labeled images and masks for mosaic propagation')
        return dict(stage=stage, **postprocess(training_images, training_masks, query, scores))
    masks = scores > .5
    confidence = ((scores < .2) | (scores > .8)).mean(axis=(1, 2))
    area = masks.sum(axis=(1, 2), dtype=np.int64)
    nonconstant = np.ptp(query, axis=(1, 2)) > 0
    selected = select_pseudo_labels(confidence, area, nonconstant)
    return dict(stage=stage, masks=masks, confidence=confidence, area=area,
                nonconstant=nonconstant, **selected)

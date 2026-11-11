"""Image-to-mosaic-to-mask stage of the corrected full TGS workflow."""
import numpy as np

from sciona.tgs_assembly import construct_mosaics
from sciona.tgs_mosaic import propagate_masks


def postprocess(training_images, training_masks, query_images, query_scores):
    """Construct mosaics jointly, then propagate eligible training masks.

    Images use population/row/column order and normalized grayscale [0,1].
    Training and query identity/disjointness are caller responsibilities. This
    transductive method intentionally uses all supplied images for assembly.
    Query scores must already be the source branch blend; this function does
    not perform model fitting or replace the full TGS execution.
    """
    train, query, masks, scores = [np.asarray(x) for x in
                                  (training_images, query_images, training_masks, query_scores)]
    for images in (train, query):
        if images.ndim != 3 or images.shape[0] < 1 or images.shape[1:] != (101, 101):
            raise ValueError('images must be nonempty 101 by 101 grayscale populations')
        if not np.isfinite(images).all() or (images < 0).any() or (images > 1).any():
            raise ValueError('images must be finite and normalized to [0,1]')
    if masks.shape != train.shape or not np.isin(masks, [0, 1]).all():
        raise ValueError('training masks must be aligned binary arrays')
    if scores.shape != query.shape or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('query scores must be aligned finite probabilities')
    population = np.concatenate((train, query)).astype(np.float64)
    grids = construct_mosaics(population.transpose(0, 2, 1))
    eligible = np.ptp(train, axis=(1, 2)) > 0
    result = propagate_masks(masks, eligible, scores > 0.5, grids)
    return dict(**result, mosaic_count=len(grids),
                mosaic_tile_count=sum(int(np.sum(grid >= 0)) for grid in grids))

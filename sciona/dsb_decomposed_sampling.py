"""Draft DSB sampling adapters using independently qualified numerical atoms.

These preserve recipe units, sample ordering and shared RNG timing. They do not
confer graph approval or make the source crop/augmentation policy generic.
Adapted from lfz/DSB2017; MIT notice: docs/licenses/DSB2017-MIT.txt.
"""
import math
import numpy as np
from sciona.atoms.numerical.statistics.threshold_sampling import (
    repeat_rows_by_threshold, sequential_softmax_indices,
)
from sciona.dsb_components import _integer, crop_classifier_proposal
from sciona.dsb_augmentation import augment_classifier


def detector_sampling_table(boxes_by_image, resolution=1.):
    """Source 1/2/4 additive size repeats, retaining image association and order."""
    if not math.isfinite(resolution) or resolution<=0:
        raise ValueError('resolution must be positive and finite')
    rows=[]
    for index,boxes in enumerate(boxes_by_image):
        boxes=np.asarray(boxes,dtype=float)
        if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.all(np.isfinite(boxes)) or np.any(boxes[:,3]<=0):
            raise ValueError('each image needs finite boxes with positive diameter')
        repeated=repeat_rows_by_threshold(boxes,3,np.array([6.,30.,40.])/resolution,[1,2,4])
        if len(repeated):rows.append(np.column_stack([np.full(len(repeated),index),repeated]))
    return np.concatenate(rows) if rows else np.empty((0,5),dtype=float)


def classifier_training_sample(volume, proposals, known_proposals, *, topk=5, crop_size=96,
                               temperature=1., scale=False, flip=True, rotate=False,
                               swap=False, rng=None):
    """Sequential sampling/crop/augmentation with a single NumPy RNG stream.

    Proposals and their indicators must already be filtered together. Returned
    zero slots and raw crop intensity conventions match source training batches.
    """
    rng = np.random.RandomState() if rng is None else rng
    proposals, known = np.asarray(proposals), np.asarray(known_proposals)
    topk, crop_size = _integer(topk,'topk'), _integer(crop_size,'crop_size')
    if (proposals.ndim != 2 or proposals.shape[1] != 5 or not np.all(np.isfinite(proposals))
            or np.any(proposals[:,4] <= 0) or known.shape != (len(proposals),)
            or not np.all((known == 0) | (known == 1)) or crop_size % 4):
        raise ValueError('invalid proposals, indicators or crop size')
    chosen = sequential_softmax_indices(proposals[:,0], topk, temperature, rng=rng)
    crops = np.zeros((topk,1,crop_size,crop_size,crop_size), dtype=np.float32)
    coords = np.zeros((topk,3,crop_size//4,crop_size//4,crop_size//4), dtype=np.float32)
    indicators = np.zeros(topk, dtype=np.float32)
    for slot,index in enumerate(chosen):
        crop,coordinate = crop_classifier_proposal(volume, proposals[index,1:], crop_size=crop_size,
                                                   phase='train', scale_enabled=scale, rng=rng)
        crop,coordinate = augment_classifier(crop,coordinate,flip=flip,rotate=rotate,swap=swap,
                                             filling_value=0,rng=rng)
        crops[slot], coords[slot], indicators[slot] = crop, coordinate, known[index]
    return crops, coords, indicators, chosen

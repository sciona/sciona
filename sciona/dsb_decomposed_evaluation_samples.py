"""Explicit DSB evaluation sampling with source policies and registered recipes."""
import numpy as np
from sciona.dsb_components import _integer
from sciona.atoms.dl.detection.dsb_spatial_recipes import crop_detection, assign_anchor_labels, crop_classifier_proposal


def prepare_classifier_batch(volume, proposals, topk=5, crop_size=96, stride=4):
    """Source test-time descending selection with zero-filled unused slots.

    Crops are raw float32 transformed intensities, without detector normalization.
    The source computes but does not apply a pad mask; unused slots are retained.
    """
    topk = _integer(topk, 'topk')
    crop_size, stride = _integer(crop_size, 'crop_size'), _integer(stride, 'stride')
    volume, proposals = np.asarray(volume), np.asarray(proposals)
    if (volume.ndim != 4 or volume.shape[0] != 1 or min(volume.shape) < 1
            or volume.dtype.kind not in 'fiu' or proposals.ndim != 2 or proposals.shape[1] != 5
            or proposals.dtype.kind != 'f' or not np.all(np.isfinite(proposals))
            or np.any(proposals[:, 4] <= 0) or crop_size % stride):
        raise ValueError('invalid classifier volume, proposals, or crop geometry')
    chosen = proposals[:, 0].argsort()[::-1][:topk]
    crops = np.zeros((topk, 1, crop_size, crop_size, crop_size), dtype=np.float32)
    coords = np.zeros((topk, 3, crop_size//stride, crop_size//stride, crop_size//stride), dtype=np.float32)
    # Test crops consume no randomness; this local stream never touches global RNG.
    rng = np.random.RandomState(0)
    for slot, index in enumerate(chosen):
        crops[slot], coords[slot] = crop_classifier_proposal(volume, proposals[index, 1:],
                                                           crop_size=crop_size, stride=stride, phase='test', rng=rng)
    return crops, coords, chosen

def classifier_validation_sample(volume,proposals,known_proposals,case_label,*,topk=5,crop_size=96):
    """Top-score crops without augmentation; retain zero slots and label alignment."""
    known=np.asarray(known_proposals)
    if known.shape!=(len(proposals),) or not np.all((known==0)|(known==1)) or case_label not in (0,1):
        raise ValueError('aligned binary proposal and case labels required')
    images,coords,chosen=prepare_classifier_batch(volume,proposals,topk,crop_size)
    indicators=np.zeros(images.shape[0],dtype=np.int32)
    indicators[:len(chosen)]=known[chosen]
    return images,coords,indicators,np.array([case_label],dtype=np.int64)

def detector_validation_sample(volume,target,boxes,*,crop_size=128,bound_size=12,
                                rng=None,label_rng=None):
    """Target-centered unscaled crop; validation anchor threshold and no subsampling."""
    sample,selected,transformed,coords=crop_detection(volume,target,boxes,
        crop_size=crop_size,bound_size=bound_size,scale=False,random_crop=False,rng=rng)
    labels=assign_anchor_labels(sample.shape[1:],selected,transformed,phase='val',rng=label_rng)
    return sample.astype(np.float32),labels,coords

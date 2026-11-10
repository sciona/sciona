"""OSIC batch assembly over an explicit, already eligible image population."""
import numpy as np

from sciona.group_member_sampling import sample_indices
from sciona.osic_image_training_controls import BATCH_SIZE


def validate_population(images, metadata, slopes):
    """Validate aligned groups of preprocessed planes and caller-owned targets.

Each image group has shape (members,512,512,1), with a shared float32/64 dtype.
Metadata has shape (groups,4), and slopes (groups,). The caller performs raw
decoding, source exclusions, longitudinal fitting and branch-specific image
normalization. No source identity list is embedded here. Failed decodes are
outside this valid-input contract, rather than silently shortening batches.
"""
    if not isinstance(images, (tuple, list)) or not images:
        raise ValueError('Nonempty ordered image groups required')
    dtype = None
    for group in images:
        if (not isinstance(group, np.ndarray) or group.ndim != 4
                or group.shape[0] == 0 or group.shape[1:] != (512,512,1)
                or group.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not np.isfinite(group).all()):
            raise ValueError('Finite full-size preprocessed image groups required')
        if dtype is not None and group.dtype != dtype:
            raise ValueError('Image groups must share a dtype')
        dtype = group.dtype
    for value, shape in ((metadata,(len(images),4)), (slopes,(len(images),))):
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype not in (np.dtype('float32'),np.dtype('float64'))
                or not np.isfinite(value).all()):
            raise ValueError('Finite aligned floating metadata and slopes required')


def sample_batch(images, metadata, slopes, random_state):
    """Draw eight groups with replacement, then one image per selected group."""
    validate_population(images, metadata, slopes)
    groups, members = sample_indices([len(group) for group in images], BATCH_SIZE, random_state)
    return [np.stack([images[g][m] for g,m in zip(groups,members)]), metadata[groups]], slopes[groups]

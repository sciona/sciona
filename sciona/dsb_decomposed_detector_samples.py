"""DSB detector sample assembly through explicit registered spatial recipes."""
import numpy as np
from sciona.atoms.dl.detection.dsb_spatial_recipes import crop_detection, augment_detection, assign_anchor_labels


def detector_indexed_training_sample(index, volumes, boxes_by_image, table, eligible_image_indices,
                                    *, rng=None, **kwargs):
    """Connect sample selection to crop/augment/labels with explicit associations."""
    from sciona.dsb_sampling import select_detector_sample
    if len(volumes)!=len(boxes_by_image):raise ValueError('volume and box collections must align')
    rng=np.random.RandomState() if rng is None else rng
    selected=select_detector_sample(index,table,eligible_image_indices,image_count=len(volumes),rng=rng)
    image=selected['image_index']
    options=dict(kwargs)
    if selected['independent_image']:options['scale']=False
    return detector_training_sample(volumes[image],selected['target'],boxes_by_image[image],
                                    random_crop=selected['random_crop'],rng=rng,**options)

def detector_training_sample(volume, target, boxes, *, crop_size=128, bound_size=12,
                             scale=True, random_crop=False, flip=True, rotate=True,
                             swap=True, num_neg=800, rng=None, label_seed=None, label_rng=None):
    """Shared NumPy RNG for crop/augment; independent Python seed for label sampling."""
    rng = np.random.RandomState() if rng is None else rng
    sample, selected, transformed_boxes, coords = crop_detection(volume, target, boxes,
        crop_size=crop_size, bound_size=bound_size, scale=scale, random_crop=random_crop, rng=rng)
    if not random_crop:
        sample, selected, transformed_boxes, coords = augment_detection(sample, selected,
            transformed_boxes, coords, flip=flip, rotate=rotate, swap=swap, rng=rng)
    labels = assign_anchor_labels(sample.shape[1:], None if np.isnan(selected[0]) else selected,
                                  transformed_boxes, num_neg=num_neg, random_state=label_seed, rng=label_rng)
    # Alternating-training loader passes raw float32 values; root detector
    # inference loader has a different normalization contract.
    return sample.astype(np.float32), labels, coords

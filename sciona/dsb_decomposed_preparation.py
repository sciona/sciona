"""DSB preparation recipes using independently reusable spatial resampling.

Source arithmetic: lfz/DSB2017; MIT notice: docs/licenses/DSB2017-MIT.txt.
Training and inference retain their distinct mask policies and annotation frames.
"""
import numpy as np
from sciona.dsb_components import segment_volume, mask_and_remove_bone
from sciona.dsb_training_preprocessing import training_lung_mask
from sciona.dsb_decomposed_annotations import training_voxel_annotations
from sciona.dsb_decomposed_annotations import world_training_annotations
from sciona.atoms.numerical.spatial.resampling import resample_regular_volume


def prepare_training_masks(volume,left_mask,right_mask,spacing,requested_spacing=(1.,1.,1.)):
    """Clean and crop already-oriented masks; preserve inputs and source bounds."""
    volume=np.asarray(volume);left_mask=np.asarray(left_mask);right_mask=np.asarray(right_mask)
    if volume.dtype.kind not in 'fi' or volume.shape!=left_mask.shape or volume.shape!=right_mask.shape:
        raise ValueError('aligned real volume and masks required')
    dilated=training_lung_mask(left_mask)|training_lung_mask(right_mask)
    mask=left_mask|right_mask
    if not mask.any():raise ValueError('cannot crop empty segmentation')
    # Modern NumPy disallows source Boolean subtraction. Dilation contains mask,
    # so its nonnegative difference is exactly the Boolean boundary below.
    boundary=dilated & ~mask
    values=volume.copy();values[np.isnan(values)]=-2000
    window=np.array([-1200.,600.])
    scaled=(values-window[0])/(window[1]-window[0])
    scaled=np.clip(scaled,0,1)
    cleaned=(scaled*255).astype(np.uint8)
    cleaned[~dilated]=170;cleaned[boundary & (cleaned>210)]=170
    resized,effective=resample_regular_volume(cleaned,spacing,requested_spacing,order=1)
    axes=np.where(mask)
    box=np.array([[a.min(),a.max()] for a in axes])
    box=np.floor(box*np.asarray(spacing)[:,None]/np.asarray(requested_spacing)[:,None]).astype(int)
    bounds=np.stack((np.maximum(0,box[:,0]-5),np.minimum(resized.shape,box[:,1]+10)),axis=1)
    cropped=resized[tuple(slice(int(lo),int(hi)) for lo,hi in bounds)]
    if min(cropped.shape)<1:raise ValueError('empty training crop')
    return cropped[None],effective,bounds

def prepare_training_volume(volume,spacing,annotations_xyz_diameter,requested_spacing=(1.,1.,1.)):
    """Segment a raw volume and connect training cleanup with voxel annotations."""
    left,right=segment_volume(volume,spacing)
    processed,effective,bounds=prepare_training_masks(volume,left,right,spacing,requested_spacing)
    boxes=training_voxel_annotations(annotations_xyz_diameter,spacing,bounds,requested_spacing)
    return processed,boxes,effective,bounds

def prepare_world_training_volume(volume,left_mask,right_mask,spacing,origin_zyx,
                                   annotations_xyz_diameter,*,flip=False,requested_spacing=(1.,1.,1.)):
    """Flip supplied volume and segmentation together, then clean and map boxes."""
    if not isinstance(flip,(bool,np.bool_)):raise ValueError('explicit Boolean flip required')
    volume=np.asarray(volume);left_mask=np.asarray(left_mask);right_mask=np.asarray(right_mask)
    if flip:
        volume=volume[:,::-1,::-1];left_mask=left_mask[:,::-1,::-1];right_mask=right_mask[:,::-1,::-1]
    processed,effective,bounds=prepare_training_masks(volume,left_mask,right_mask,spacing,requested_spacing)
    boxes=world_training_annotations(annotations_xyz_diameter,origin_zyx,spacing,bounds,volume.shape,
                                     flip=flip,requested_spacing=requested_spacing)
    return processed,boxes,effective,bounds

def preprocess_volume(volume, spacing, requested_spacing=(1., 1., 1.)):
    """Connected source preprocessing from runtime array through segmentation/crop."""
    first, second = segment_volume(volume, spacing)
    return preprocess_from_masks(volume, first, second, spacing, requested_spacing)

def preprocess_from_masks(volume, left_mask, right_mask, spacing, requested_spacing=(1., 1., 1.)):
    """Compose source mask cleanup, resampling, and asymmetric bounding-box crop.

    Returns cropped uint8 volume, effective spacing, and exclusive crop bounds
    in the resampled grid. Source crop bounds use requested spacing, even where
    rounding changes effective spacing; both are retained for downstream mapping.
    Segmentation must supply the masks; an empty union is rejected explicitly.
    """
    cleaned = mask_and_remove_bone(volume, left_mask, right_mask)
    union = np.asarray(left_mask) | np.asarray(right_mask)
    if not union.any():
        raise ValueError('cannot crop an empty segmentation')
    resampled, effective_spacing = resample_regular_volume(cleaned, spacing, requested_spacing, order=1)
    coordinates = np.where(union)
    box = np.array([[axis.min(), axis.max()] for axis in coordinates])
    box = np.floor(box * np.asarray(spacing)[:, None] / np.asarray(requested_spacing)[:, None]).astype(int)
    bounds = np.stack([np.maximum(0, box[:, 0] - 5),
                       np.minimum(resampled.shape, box[:, 1] + 10)], axis=1)
    crop = resampled[tuple(slice(int(lo), int(hi)) for lo, hi in bounds)]
    if min(crop.shape) < 1:
        raise ValueError('source crop bounds produce an empty volume')
    return crop, effective_spacing, bounds

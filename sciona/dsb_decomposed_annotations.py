"""DSB annotation policies composed with generic ordered coordinate scaling.

Source arithmetic: lfz/DSB2017; MIT notice: docs/licenses/DSB2017-MIT.txt.
"""
import numpy as np
from sciona.dsb_components import _integer
from sciona.atoms.numerical.spatial.coordinate_scaling import scale_coordinate_rows


def training_voxel_annotations(annotations_xyz_diameter,spacing,crop_bounds,requested_spacing=(1.,1.,1.)):
    """Source XYZ voxel annotations -> cropped ZYX boxes; diameter uses Y spacing.

    Empty and first-zero-Z source sentinels become loader-ready (0,4) arrays.
    No identifier columns enter this numerical boundary.
    """
    rows=np.asarray(annotations_xyz_diameter,dtype=float)
    spacing=np.asarray(spacing,dtype=float);resolution=np.asarray(requested_spacing,dtype=float)
    bounds=np.asarray(crop_bounds)
    if (rows.ndim!=2 or rows.shape[1]!=4 or not np.all(np.isfinite(rows))
            or spacing.shape!=(3,) or resolution.shape!=(3,) or bounds.shape!=(3,2)
            or not np.all(np.isfinite(spacing)) or not np.all(np.isfinite(resolution))
            or np.any(spacing<=0) or np.any(resolution<=0) or not np.all(np.isfinite(bounds))):
        raise ValueError('finite annotation rows and spatial transform required')
    if not len(rows) or rows[0,2]==0:return np.empty((0,4),dtype=float)
    if np.any(rows[:,3]<=0):raise ValueError('positive annotation diameters required')
    return scale_coordinate_rows(rows[:,[2,1,0,3]], np.r_[spacing,spacing[1]],
                                 np.r_[resolution,resolution[1]], np.r_[bounds[:,0],0])

def world_training_annotations(annotations_xyz_diameter,origin_zyx,spacing,crop_bounds,
                                volume_shape,*,flip=False,requested_spacing=(1.,1.,1.)):
    """Source absolute world displacement, optional size-coordinate flip, and crop.

    This path has no first-zero-Z sentinel: a world annotation on the origin plane
    remains valid. Empty arrays represent absence of annotations explicitly.
    """
    rows=np.asarray(annotations_xyz_diameter,dtype=float)
    origin=np.asarray(origin_zyx,dtype=float);spacing=np.asarray(spacing,dtype=float)
    resolution=np.asarray(requested_spacing,dtype=float);bounds=np.asarray(crop_bounds,dtype=float)
    shape=np.array([_integer(n,'volume dimension') for n in volume_shape])
    if (rows.ndim!=2 or rows.shape[1]!=4 or not np.all(np.isfinite(rows)) or np.any(rows[:,3]<=0)
            or origin.shape!=(3,) or spacing.shape!=(3,) or resolution.shape!=(3,) or shape.shape!=(3,)
            or bounds.shape!=(3,2) or not all(np.all(np.isfinite(a)) for a in [origin,spacing,resolution,bounds])
            or np.any(spacing<=0) or np.any(resolution<=0) or not isinstance(flip,(bool,np.bool_))):
        raise ValueError('finite world annotations and explicit spatial transform required')
    labels=[]
    for row in rows:
        position=np.absolute(row[:3][::-1]-origin)/spacing
        if flip:position[1:]=shape[1:]-position[1:]
        labels.append(np.concatenate([position,[row[3]/spacing[1]]]))
    if not labels:return np.empty((0,4),dtype=float)
    return scale_coordinate_rows(np.array(labels), np.r_[spacing,spacing[1]],
                                 np.r_[resolution,resolution[1]], np.r_[bounds[:,0],0])

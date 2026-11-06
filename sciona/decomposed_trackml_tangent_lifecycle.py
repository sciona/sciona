"""Bind reusable tangent geometry with source regulation kept in the recipe."""
import numpy as np

from sciona.trackml_helix import _vectors
from sciona.trackml_nearest import _parameter
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_direction_lifecycle import execute as direction_execute
from sciona.decomposed_trackml_residual_lifecycle import residual_class


def construction(x, y, z, ux, uy, uz, pitch, *, operation,
                 min_uz0=.001, max_mg_over_qB=1e6):
    values = _vectors([x,y,z,ux,uy,uz])
    h = np.broadcast_to(_parameter(pitch,values[0].shape),values[0].shape)
    center, radius = operation(np.column_stack(values[:3]),
                               np.column_stack(values[3:]),h,
                               min_uz0,max_mg_over_qB)
    center = np.asarray(center); radius = np.asarray(radius)
    n = len(h)
    if (center.dtype != np.float64 or center.shape != (n,2)
            or radius.dtype != np.float64 or radius.shape != (n,)
            or not np.isfinite(center).all() or not np.isfinite(radius).all()
            or np.any(radius < 0)):
        raise ValueError('Finite aligned centers and nonnegative radii required')
    return center[:,0],center[:,1],radius


def tangent(x, y, z, uz, xm, ym, radius, pitch, *, operation):
    x,y,z,uz,xm,ym,radius,pitch = _vectors([x,y,z,uz,xm,ym,radius,pitch])
    result = np.asarray(operation(np.column_stack((x,y,z)),
                                  np.column_stack((xm,ym)),radius,pitch,uz))
    if result.dtype != np.float64 or result.shape != (len(x),3) or not np.isfinite(result).all():
        raise ValueError('Finite aligned tangent matrix required')
    return result


def execute(*args, construction_function, tangent_function, **kwargs):
    if not callable(construction_function) or not callable(tangent_function):
        raise ValueError('Callable tangent geometry providers required')
    def local_class(base, displacement_function, uncertainty_function):
        parent = residual_class(base,displacement_function,uncertainty_function)
        class TangentGeometry(parent):
            pass
        def construct(*values, **controls):
            return construction(*values,operation=construction_function,**controls)
        def orient(*values):
            return tangent(*values,operation=tangent_function)
        TangentGeometry.chooseLikelySecondHits = _bound(
            parent.chooseLikelySecondHits, {'helixWithTangentVector': construct})
        for name in ('dropNeighborsInconsistentWithCellFeatures','evaluateCellFeatureConsistency'):
            setattr(TangentGeometry,name,_bound(getattr(parent,name),
                    {'helixUnitTangentVector': orient}))
        return TangentGeometry
    return _bound(direction_execute, {'residual_class': local_class})(*args,**kwargs)

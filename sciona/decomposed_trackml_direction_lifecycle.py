"""Bind reusable helix direction while preserving invocation-local recipe policy."""
import numpy as np
from sciona.trackml_helix import _vectors
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_grid_lifecycle import execute as grid_execute
from sciona.decomposed_trackml_residual_lifecycle import execute as residual_execute, residual_class


def direction(hel_xm, hel_ym, hel_pitch, x1, y1, z1, x2, y2, z2, *, operation):
    if not callable(operation):
        raise ValueError('Callable direction provider required')
    arrays = _vectors([hel_xm,hel_ym,hel_pitch,x1,y1,z1,x2,y2,z2])
    if not arrays[0].size:
        return np.empty(0,dtype=np.float64)
    xm,ym,pitch,ax,ay,az,bx,by,bz = arrays
    # Reject hidden overflow before sign() can turn an infinite intermediate
    # into a finite indicator. Preserve the retained arithmetic order.
    try:
        with np.errstate(over='raise',invalid='raise'):
            dz = bz-az
            mask = dz == 0
            cross = ((ay[mask]-by[mask])*(xm[mask]-ax[mask])
                     +(bx[mask]-ax[mask])*(ym[mask]-ay[mask]))
            if not np.isfinite(cross).all():
                raise ValueError('Nonfinite direction arithmetic')
            result = np.asarray(operation(*arrays))
    except FloatingPointError as exc:
        raise ValueError('Nonfinite direction arithmetic') from exc
    if result.dtype != np.float64 or result.shape != arrays[0].shape or not np.isfinite(result).all():
        raise ValueError('Finite aligned float64 direction output required')
    return result


def execute(*args, direction_function, **kwargs):
    def local_class(base, displacement_function, uncertainty_function):
        parent = residual_class(base,displacement_function,uncertainty_function)
        class Directed(parent):
            pass
        def operation(*values):
            return direction(*values,operation=direction_function)
        Directed.getHelixParams = _bound(parent.getHelixParams, {'helixDirectionFromTwoPoints': operation})
        return Directed
    residual = _bound(residual_execute, {'residual_class': local_class})
    return _bound(grid_execute, {'residual_execute': residual})(*args,**kwargs)

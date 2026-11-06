"""Source-specific attack adapter composed from reusable row normalization.

Derived from dongyp13 non-targeted commit5c68162e05de7afed2e3d33115df43e2a7a1c3da
and targeted commit7da707fe568053a9e2735fcc692c8f0ad7e122a1. Apache-2.0;
see docs/licenses/Adversarial-*-Apache-2.0.txt. Neural gradient computation is separate.
"""
import math
import numpy as np
from sciona.atoms.numerical.statistics.row_scale import normalize_rows_by_scale
from sciona.adversarial_updates import configuration


def update(x, gradient, accumulated, lower, upper, *, mode, epsilon, momentum=1.,
           non_targeted_iterations=10):
    """Return projected image and next momentum, without mutating input arrays.

    Values are NHWC float32/64 in normalized [-1,1]. Zero normalization
    denominators explicitly fail instead of silently propagating source NaNs;
    no stabilizing epsilon or substituted update is introduced.
    """
    config=configuration(mode,epsilon,non_targeted_iterations=non_targeted_iterations)
    if type(momentum) not in (float,int) or not math.isfinite(momentum) or momentum<0:
        raise ValueError('momentum must be finite and nonnegative')
    arrays=(x,gradient,accumulated,lower,upper)
    if (not isinstance(x,np.ndarray) or x.ndim!=4 or min(x.shape)==0 or x.shape[-1]!=3
            or x.dtype not in (np.dtype('float32'),np.dtype('float64'))
            or any(not isinstance(a,np.ndarray) or a.shape!=x.shape or a.dtype!=x.dtype
                   or not np.isfinite(a).all() for a in arrays)):
        raise ValueError('update requires matching finite NHWC floating arrays')
    if np.any(lower < -1) or np.any(upper > 1) or np.any(lower>x) or np.any(x>upper):
        raise ValueError('image must lie inside valid normalized clipping bounds')
    # The neural boundary supplies contiguous NHWC arrays. Restrict this
    # adapter to that qualified layout rather than changing reduction order.
    if any(not a.flags.c_contiguous for a in arrays):
        raise ValueError('contiguous NHWC arrays required')
    statistic='mean_absolute' if mode=='non_targeted' else 'population_std'
    scaled=normalize_rows_by_scale(gradient.reshape(len(x),-1),statistic).reshape(x.shape)
    noise=momentum*accumulated+scaled
    if mode=='targeted':
        noise=normalize_rows_by_scale(noise.reshape(len(x),-1),'population_std').reshape(x.shape)
        result=x-config['alpha']*np.clip(np.round(noise),-2,2)
    else:
        result=x+config['alpha']*np.sign(noise)
    result=np.clip(result,lower,upper)
    if not np.isfinite(result).all() or not np.isfinite(noise).all():
        raise ValueError('update became nonfinite')
    return result,noise

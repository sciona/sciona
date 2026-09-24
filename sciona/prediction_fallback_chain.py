"""Bounded prediction fallbacks without swallowing process-control exceptions."""
import numpy as np


def prediction_fallback_chain(primary, baseline, count, constant):
    """Return (owned prediction vector, route) from primary/baseline/constant.

    Callbacks own their input-isolation policy. Ordinary callback failures or
    invalid output vectors select the next route. KeyboardInterrupt, SystemExit
    and other BaseException subclasses propagate. Returned arrays are copied;
    no callback-owned output is mutated. Empty queries do not invoke callbacks.
    """
    if not callable(primary) or not callable(baseline) or type(count) is not int or count<0:
        raise ValueError('Callable predictors and nonnegative integer count required')
    if type(constant) not in (int,float) or not np.isfinite(constant):
        raise ValueError('Explicit finite constant fallback required')
    if count==0:return np.empty(0,dtype=np.float64),'empty'
    for name,predict in [('primary',primary),('baseline',baseline)]:
        try:
            result=predict()
            if (not isinstance(result,np.ndarray) or result.shape!=(count,)
                    or result.dtype.kind not in 'iuf' or not np.isfinite(result).all()):
                raise ValueError('Finite aligned prediction vector required')
            return result.copy(),name
        except Exception:
            continue
    return np.full(count,float(constant)),'constant'

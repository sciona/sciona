"""Reusable numeric boundaries for the retained independent VSB feature recipe.

No grouping identities, labels, or measurement-triple assumptions enter these
operations. The recipe binds its own retention, smoothing and knee controls.
"""
import numpy as np
from .vsb_peaks import _signal,plateau_location


def baseline_arguments(values,retention):
    """Construct coefficients and initial delay for an external linear filter."""
    x=_signal(values)
    if isinstance(retention,(bool,np.bool_)) or not isinstance(retention,(int,float,np.integer,np.floating)) or not np.isfinite(retention) or not 0<=retention<=1:
        raise ValueError('Retention must lie between zero and one')
    a=float(retention)
    return np.array([a,-a]),np.array([1.,-a]),x,-1,np.array([-a*x[0]])


def baseline_residual(filtered):
    """Validate filtered samples and enforce the initial zero residual boundary.

    This does not authenticate filter coefficients or same-invocation provenance.
    The graph must bind the exact filter and the matching argument outputs.
    """
    result=_signal(filtered)
    result[0]=0.
    return result


def rank_peaks(residual,positions):
    """Rank supplied candidate positions by descending absolute amplitude."""
    x=_signal(residual);p=np.asarray(positions)
    if p.ndim!=1 or p.dtype.kind not in 'iu' or len(p)<2 or np.any(p<0) or np.any(p>=len(x)) or len(np.unique(p))!=len(p):
        raise ValueError('At least two unique in-range integer candidates required')
    heights=np.abs(x[p]);order=np.argsort(heights)[::-1]
    return p[order].copy(),heights[order]


def amplitude_gradient(heights,width,trim):
    """Smooth adjacent amplitude differences, then trim both convolution ends.

    Full convolution and source slicing order are intentional. An empty trimmed
    result is valid and drives the source knee fallback. Controls are explicit.
    """
    h=_signal(heights)
    if len(h)<2 or np.any(h<0) or np.any(np.diff(h)>0):
        raise ValueError('At least two nonnegative descending amplitudes required')
    if type(width) is not int or width<1 or type(trim) is not int or trim<0:
        raise ValueError('Positive integer width and nonnegative integer trim required')
    result=np.convolve(np.diff(h),np.ones(width)/width)
    result=result[trim:-trim] if trim else result
    if not np.isfinite(result).all():raise ValueError('Gradient arithmetic overflow')
    return result


def select_ranked_peaks(positions,heights,gradient,threshold,count,offset):
    """Apply the cumulative threshold knee and return chronological selections.

    Counts need not be consecutive. Preserve negative Python slice endpoints;
    clamping a negative knee would change the qualified source computation.
    """
    p=np.asarray(positions);h=_signal(heights)
    if p.ndim!=1 or p.dtype.kind not in 'iu' or len(p)!=len(h) or len(p)<2 or np.any(p<0) or len(np.unique(p))!=len(p):
        raise ValueError('Aligned unique nonnegative integer candidates required')
    if np.any(h<0) or np.any(np.diff(h)>0):raise ValueError('Descending nonnegative amplitudes required')
    if type(offset) is not int or offset<0:raise ValueError('Nonnegative integer knee offset required')
    knee=plateau_location(gradient,threshold=threshold,count=count)-offset
    chosen=p[:knee];amplitudes=h[:knee];order=np.argsort(chosen)
    return chosen[order].copy(),amplitudes[order].copy(),knee

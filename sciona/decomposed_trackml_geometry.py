"""Source-compatible boundaries for separately reusable numeric providers.

No detector identifiers, field strengths, units or tracking policy are encoded.
Empty internal batches bypass providers whose public contract requires samples.
Undefined source geometry rejects rather than accepting substituted finite values.
Phase and degeneracy guards derive from the BSD-2-Clause TrackML source; see
docs/licenses/TrackML-BSD-2-Clause.txt.
"""
import numpy as np
from sciona.trackml_helix import _vectors, _replacement, _evaluate


def circle(x1,y1,x2,y2,x3,y3,large_radius,operation):
    values=_vectors([x1,y1,x2,y2,x3,y3])
    if isinstance(large_radius,bool) or not np.isscalar(large_radius) or not np.isfinite(large_radius) or large_radius<=0:
        raise ValueError('Positive finite large radius required')
    if not values[0].size:return tuple(np.empty(0,dtype=np.float64) for _ in range(3))
    x1,y1,x2,y2,x3,y3=values
    same21=(x2==x1)&(y2==y1);same31=(x3==x1)&(y3==y1)
    xd=np.where(same21&same31,x1,np.where(same21,x3-x1,x2-x1))
    yd=np.where(same21&same31,y1,np.where(same21,y3-y1,y2-y1))
    if np.any(np.square(xd)+np.square(yd)==0):
        raise ValueError('Zero tangent magnitude has no source-defined fallback')
    return _evaluate(operation,values,large_radius=np.float64(large_radius))


def pitch_two(x1,y1,z1,x2,y2,z2,xm,ym,zero_pitch,operation):
    values=_vectors([x1,y1,z1,x2,y2,z2,xm,ym]);replacement=_replacement(zero_pitch)
    if not values[0].size:return tuple(np.empty(0,dtype=np.float64) for _ in range(3))
    return _evaluate(operation,values,zero_pitch=replacement)


def pitch_three(x1,y1,z1,x2,y2,z2,x3,y3,z3,xm,ym,zero_pitch,operation):
    values=_vectors([x1,y1,z1,x2,y2,z2,x3,y3,z3,xm,ym]);replacement=_replacement(zero_pitch)
    if not values[0].size:return tuple(np.empty(0,dtype=np.float64) for _ in range(4))
    x1,y1,z1,x2,y2,z2,x3,y3,z3,xm,ym=values
    phi1=np.arctan2(y1-ym,x1-xm)
    a=np.arctan2(y2-ym,x2-xm)-phi1
    b=np.arctan2(y3-ym,x3-xm)-phi1
    a[a>=np.pi]-=2*np.pi;a[a < -np.pi]+=2*np.pi
    negative=a<0
    b[~negative & (b<a)]+=2*np.pi;b[negative & (b>a)]-=2*np.pi
    # Match the source's exact floating-point denominator, including cancellation.
    if np.any(3.0*(np.square(a)+np.square(b))-np.square(a+b)==0):
        raise ValueError('Zero phase variance has no source-defined regression')
    return _evaluate(operation,values,zero_pitch=replacement)


def bind(circle_function,pitch_two_function,pitch_three_function):
    """Bind explicit numeric providers to source-compatible invocation boundaries."""
    from functools import partial
    if not all(callable(f) for f in (circle_function,pitch_two_function,pitch_three_function)):
        raise ValueError('Three callable geometry providers required')
    return dict(circle_operation=partial(circle,large_radius=1e6,operation=circle_function),
                pitch_two_operation=partial(pitch_two,zero_pitch=.001,operation=pitch_two_function),
                pitch_three_operation=partial(pitch_three,zero_pitch=.001,operation=pitch_three_function))

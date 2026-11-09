"""Domain-independent blending over an explicit directed neighborhood mask."""
import numpy as np


def blend_neighbor_means(values, neighbors, strength):
    """Blend each value with its selected neighbors' unweighted mean.

    A boolean row selects neighbors for that output. Empty rows retain their
    original value. Self-neighbors are allowed when explicitly selected. Inputs
    may be signed; this operation does not clip, infer adjacency, or impose units.
    Directed/unequal-degree masks do not generally preserve the global sum.
    """
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in 'iuf':
        raise ValueError('Nonempty real numeric vector required')
    with np.errstate(over='raise', invalid='raise'):
        try:
            vector = raw.astype(np.float64, copy=True)
        except FloatingPointError as error:
            raise ValueError('Values must be representable as finite float64') from error
    if not np.isfinite(vector).all():
        raise ValueError('Finite values required')
    mask = np.asarray(neighbors)
    if mask.dtype.kind != 'b' or mask.shape != (len(vector),len(vector)):
        raise ValueError('Aligned square boolean neighbor mask required')
    if type(strength) not in (int,float) or not 0 <= strength <= 1:
        raise ValueError('Finite blend strength in [0,1] required')
    result = vector.copy()
    if strength == 0:
        return result
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for i,row in enumerate(mask):
                count = int(row.sum())
                if count:
                    # Scale before summation to avoid avoidable neighbor-sum overflow.
                    mean = np.sum(vector[row]/count)
                    result[i] = (1-strength)*vector[i]+strength*mean
        except FloatingPointError as error:
            raise ValueError('Neighbor blend is not representable as finite float64') from error
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite neighbor blend')
    return result

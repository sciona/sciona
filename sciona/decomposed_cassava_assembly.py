"""Source identity/policy adapters for two explicit reusable ordered sums.

The first sum adds ViT and ResNeXt before scaling. Combining all four rows
with weights [.5,.5,1,1] changes subnormal rounding and is not equivalent.
No adapter calls a registered numerical provider internally.
"""
import numpy as np

from sciona.cassava_fold_contract import _keys


def pair_inputs(expected_keys, *, vit, resnext, efficientnet, cropnet):
    """Align all family identities and validate source probability contracts."""
    expected = _keys(expected_keys)
    aligned = []
    for entry in (vit, resnext, efficientnet, cropnet):
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError('Each family must supply identities and probabilities')
        keys, values = entry
        keys = _keys(keys)
        values = np.asarray(values)
        if set(keys) != set(expected) or values.shape != (len(keys), 5):
            raise ValueError('Every family must cover exactly the requested identities')
        positions = {key: i for i, key in enumerate(keys)}
        aligned.append(values[[positions[key] for key in expected]])
    for array in aligned:
        if (array.dtype.kind not in 'iuf' or not np.isfinite(array).all()
                or (array < 0).any() or (array > 1).any()
                or not np.allclose(array.sum(axis=1), 1., rtol=1e-6, atol=1e-7)):
            raise ValueError('Aligned finite normalized family probabilities required')
    a, b, c, d = [array.astype(float) for array in aligned]
    context = dict(shape=a.shape, efficientnet=c.reshape(-1), cropnet=d.reshape(-1))
    return context, np.stack([a.reshape(-1), b.reshape(-1)]), np.array([1., 1.])


def _context(context):
    if not isinstance(context, dict) or set(context) != {'shape', 'efficientnet', 'cropnet'}:
        raise ValueError('Complete same-invocation assembly context required')
    shape = context['shape']
    if (not isinstance(shape, tuple) or len(shape) != 2 or any(type(n) is not int for n in shape)
            or shape[0] < 1 or shape[1] != 5):
        raise ValueError('Nonempty five-class shape required')
    for name in ('efficientnet', 'cropnet'):
        _vector(context[name], shape[0]*5, 1.)
    return shape


def _vector(values, size, upper):
    if (not isinstance(values, np.ndarray) or values.shape != (size,)
            or values.dtype != np.dtype('float64') or not np.isfinite(values).all()
            or (values < 0).any() or (values > upper).any()):
        raise ValueError('Aligned finite float64 assembly vector within source bounds required')


def final_inputs(context, paired):
    """Supply half of the completed pair sum, then B4 and CropNet in order."""
    shape = _context(context)
    _vector(paired, shape[0]*5, 2.)
    return np.stack([paired, context['efficientnet'], context['cropnet']]), np.array([.5, 1., 1.])


def result(context, scores):
    """Restore the source matrix and argmax labels; no training provenance claim."""
    shape = _context(context)
    _vector(scores, shape[0]*5, 3.)
    matrix = scores.reshape(shape).copy()
    return dict(scores=matrix, labels=np.argmax(matrix, axis=1))

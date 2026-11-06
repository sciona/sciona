"""Four-output nearest-helix boundary for explicit reusable providers.

The tracking recipe uses the fixed-iteration source approximation, with no
claim of global minimization. Optional phase/arc outputs are outside this
boundary; no phase is reconstructed from rounded coordinates.
"""
import numpy as np
from sciona.trackml_helix import _vectors, _evaluate
from sciona.trackml_nearest import _parameter


def nearest(x0, y0, z0, hel_xm, hel_ym, hel_r, hel_pitch, x, y, z,
            iterations=3, *, operation):
    """Preserve empty batches and reject nonfinite input/output geometry."""
    if not callable(operation):
        raise ValueError('Callable nearest-point provider required')
    targets = _vectors([x, y, z])
    parameters = [_parameter(v, targets[0].shape)
                  for v in (x0, y0, z0, hel_xm, hel_ym, hel_r, hel_pitch)]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError('Positive integer Newton iteration count required')
    if not targets[0].size:
        return tuple(np.empty(0, dtype=np.float64) for _ in range(4))
    return _evaluate(operation, [*parameters, *targets], iterations=iterations)

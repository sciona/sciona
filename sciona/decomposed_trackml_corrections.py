"""Correction policy around reusable numeric atoms; BSD-2-Clause source adaptation.

See docs/licenses/TrackML-BSD-2-Clause.txt. Caller-owned calibration lookup,
missing-coefficient policy, detector bounds and in-place updates stay here.
"""
import numpy as np
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.trackml_corrections import HelixCorrector


def _domain(arrays, *, radius=None, pitch=None):
    if any(a.dtype != np.float64 or a.ndim != 1 or a.shape != arrays[0].shape
           or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Aligned finite float64 correction vectors required')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        transverse = np.sqrt(np.square(arrays[0])+np.square(arrays[1]))
        if np.any(transverse <= 0) or not np.isfinite(transverse).all():
            raise ValueError('Finite nonzero transverse radius required')
        if radius is None:
            denominator = np.abs(pitch/(2*np.pi))
        else:
            if np.any(radius <= 0): raise ValueError('Positive helix radius required')
            denominator = (np.square(radius)+np.square(np.abs(pitch/(2*np.pi))))/radius
        if np.any(denominator <= 0) or not np.isfinite(denominator).all():
            raise ValueError('Finite positive correction denominator required')


def corrector_class(base, cap_function, cylinder_function):
    if not callable(cap_function) or not callable(cylinder_function):
        raise ValueError('Two callable correction providers required')

    class Corrector(base):
        def correctCapIntersections(self, xi, yi, zi, cap_id):
            if self._run.layer_functions is not None and 'dp0' in self._run.layer_functions.functions:
                radial = np.sqrt(np.square(xi)+np.square(yi))
                phi = np.arctan2(yi, xi)
                direction, _, _, _, pitch = self.helixParams()
                charge = -np.sign(direction*pitch)
                points = np.stack([self._hel_points[-1][2], phi, radial], axis=1)
                first, second = self._run.neighbors.evaluateLayerFunctions(
                    None, None, None, False, cap_id, functions=('dp0','dp1'), points=points)
                first[np.isnan(first)] = 0
                second[np.isnan(second)] = 0
                beyond = ((radial > self._algo.spec.caps.caps_r2_max[cap_id]+10.) |
                          (radial < self._algo.spec.caps.caps_r2_min[cap_id]-10.))
                selected = ~beyond
                if selected.any():
                    args = [a[selected] for a in (xi, yi, pitch, charge, first, second)]
                    _domain(args, pitch=args[2])
                    x, y = cap_function(*args)
                    xi[selected], yi[selected] = x, y
            return xi, yi

        def correctCylinderIntersections(self, xi, yi, zi, cyl_id, mask):
            if self._run.layer_functions is not None and 'dp0' in self._run.layer_functions.functions:
                phi = np.arctan2(yi, xi)
                direction, _, _, radius, pitch = self.helixParams()
                charge = -np.sign(direction*pitch)
                last_radius = np.sqrt(np.square(self._hel_points[-1][0])+np.square(self._hel_points[-1][1]))
                points = np.stack([last_radius, zi, phi], axis=1)
                first, second = self._run.neighbors.evaluateLayerFunctions(
                    None, None, None, True, cyl_id, functions=('dp0','dp1'), points=points)
                first[np.isnan(first)] = 0
                second[np.isnan(second)] = 0
                beyond = ((zi > self._algo.spec.cylinders.cyl_absz_max[cyl_id]+10.) |
                          (zi < -self._algo.spec.cylinders.cyl_absz_max[cyl_id]-10.))
                selected = np.asarray(mask, dtype=bool) & ~beyond
                if selected.any():
                    args = [a[selected] for a in (xi, yi, zi, radius, pitch, charge, first, second)]
                    _domain(args, radius=args[3], pitch=args[4])
                    x, y, z = cylinder_function(*args)
                    xi[selected], yi[selected], zi[selected] = x, y, z
            return xi, yi, zi

    for name in ('clone', 'subset'):
        setattr(Corrector, name, _bound(getattr(HelixCorrector, name), {'HelixCorrector': Corrector}))
    return Corrector

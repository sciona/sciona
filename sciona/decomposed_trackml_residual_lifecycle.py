"""Inject reusable residual arithmetic throughout the retained tracking recipe.

BSD-2-Clause adaptation; see docs/licenses/TrackML-BSD-2-Clause.txt.
Empirical uncertainty coefficients belong to this recipe, not the providers.
"""
import numpy as np

from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_correction_lifecycle import lifecycle_class
from sciona.decomposed_trackml_commit_lifecycle import execute as commit_execute
from sciona.decomposed_trackml_detector_lifecycle import execute as detector_execute


def residual_class(base, displacement_function, uncertainty_function):
    if not all(callable(fn) for fn in (displacement_function, uncertainty_function)):
        raise ValueError('Two callable residual providers required')

    class Residuals(base):
        def projectNeighborDisplacement(self, x, y, z, xn, yn, zn):
            return displacement_function(np.column_stack((x, y, z)),
                                         np.column_stack((xn, yn, zn)))

        def predictRandomError(self, run, hel_s, e_theta_meas, e_phi_meas):
            driver = np.asarray(hel_s)
            return (uncertainty_function(np.asarray(e_theta_meas), driver, .007884, .2096),
                    uncertainty_function(np.asarray(e_phi_meas), driver, .007884, .3853))

    return Residuals


def execute(*args, displacement_function, uncertainty_function, **kwargs):
    def local_class(*providers):
        return residual_class(lifecycle_class(*providers), displacement_function, uncertainty_function)
    commit = _bound(commit_execute, {'lifecycle_class': local_class})
    return _bound(detector_execute, {'commit_execute': commit})(*args, **kwargs)

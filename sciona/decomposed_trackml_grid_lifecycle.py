"""Inject generic grid interpolation while retaining detector coordinate policy.

BSD-2-Clause adaptation; see docs/licenses/TrackML-BSD-2-Clause.txt.
Singleton removal, missing maps and coordinate dispatch remain source recipe.
"""
import numpy as np
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_neighbor_lifecycle import neighbor_class, execute as neighbor_execute
from sciona.decomposed_trackml_detector_lifecycle import execute as detector_execute
from sciona.decomposed_trackml_residual_lifecycle import execute as residual_execute
from sciona.trackml_layer_functions import createLayerFunctionInterpolators


def interpolator_factory(fit_function, evaluate_function):
    if not all(callable(fn) for fn in (fit_function, evaluate_function)):
        raise ValueError('Two callable grid providers required')

    class Interpolator:
        def __init__(self, *, points, values, method, bounds_error, fill_value):
            if method != 'linear' or bounds_error is not False or fill_value is not None:
                raise ValueError('Source linear extrapolation configuration required')
            self.dimensions = len(points)
            self.state = fit_function(points, values)

        def __call__(self, points):
            query = np.asarray(points)
            if query.ndim == 1 and self.dimensions == 1:
                query = query[:, None]
            return evaluate_function(query, self.state)

    return _bound(createLayerFunctionInterpolators, {'RegularGridInterpolator': Interpolator})


def grid_neighbor_class(base, factory):
    cylinders = base.__init__.__globals__['CylinderNeighbors']
    caps = base.__init__.__globals__['CapNeighbors']

    class Cylinder(cylinders.Cylinder):
        pass

    class Cap(caps.Cap):
        pass

    Cylinder.__init__ = _bound(cylinders.Cylinder.__init__, {'createLayerFunctionInterpolators': factory})
    Cap.__init__ = _bound(caps.Cap.__init__, {'createLayerFunctionInterpolators': factory})

    class Cylinders(cylinders):
        pass

    class Caps(caps):
        pass

    Cylinders.Cylinder, Caps.Cap = Cylinder, Cap

    class Neighbors(base):
        pass

    Neighbors.__init__ = _bound(base.__init__, {'CylinderNeighbors': Cylinders, 'CapNeighbors': Caps})
    return Neighbors


def execute(*args, grid_fit_function, grid_evaluate_function, **kwargs):
    factory = interpolator_factory(grid_fit_function, grid_evaluate_function)
    def local_neighbors(*providers):
        return grid_neighbor_class(neighbor_class(*providers), factory)
    neighbors = _bound(neighbor_execute, {'neighbor_class': local_neighbors})
    detector = _bound(detector_execute, {'neighbor_execute': neighbors})
    return _bound(residual_execute, {'detector_execute': detector})(*args, **kwargs)

"""Invocation-local fitted search providers in the complete tracking lifecycle."""
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_correction_lifecycle import lifecycle_class as correction_class
from sciona.decomposed_trackml_intersections import intersector_class
from sciona.trackml_execution import execute_tracking


def neighbor_class(base, fit_function, k_function, radius_function):
    if not all(callable(fn) for fn in (fit_function, k_function, radius_function)):
        raise ValueError('Three callable fitted search providers required')

    class Search:
        def fit(self, points):
            self.state = fit_function(points)
            return self

        def kneighbors(self, points, n_neighbors):
            return k_function(points, self.state, n_neighbors)

        def radius_neighbors(self, points, radius):
            return radius_function(points, self.state, radius)

    cylinders = base.__init__.__globals__['CylinderNeighbors']
    caps = base.__init__.__globals__['CapNeighbors']

    class Cylinder(cylinders.Cylinder):
        pass

    class Cap(caps.Cap):
        pass

    Cylinder.__init__ = _bound(cylinders.Cylinder.__init__, {'NearestNeighbors': Search})
    Cap.__init__ = _bound(caps.Cap.__init__, {'NearestNeighbors': Search})

    class Cylinders(cylinders):
        pass

    class Caps(caps):
        pass

    Cylinders.Cylinder, Caps.Cap = Cylinder, Cap

    class Neighbors(base):
        pass

    Neighbors.__init__ = _bound(base.__init__, {'CylinderNeighbors': Cylinders, 'CapNeighbors': Caps})
    return Neighbors


def execute(modules, hits, circle_function, pitch_two_function, pitch_three_function,
            nearest_function, cylinder_function, plane_function,
            cylinder_intersection_function, plane_intersection_function,
            cap_correction_function, cylinder_correction_function,
            search_fit_function, search_k_function, search_radius_function,
            *, cells=None, layer_functions=None, params=None, event_id=0):
    base = correction_class(circle_function, pitch_two_function, pitch_three_function,
                            nearest_function, cylinder_function, plane_function,
                            cap_correction_function, cylinder_correction_function)
    neighbors = neighbor_class(base.Run.__init__.__globals__['Neighbors'],
                               search_fit_function, search_k_function, search_radius_function)

    class Run(base.Run):
        pass

    Run.__init__ = _bound(base.Run.__init__, {'Neighbors': neighbors})

    class Lifecycle(base):
        pass

    Lifecycle.Run = Run
    intersector = intersector_class(cylinder_intersection_function, plane_intersection_function)
    run = _bound(execute_tracking, {'CheckedLifecycle': Lifecycle, 'Intersector': intersector})
    return run(modules, hits, cells=cells, layer_functions=layer_functions, params=params, event_id=event_id)

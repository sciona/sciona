"""Invocation-local coordinate projection providers in every tracking round."""
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_projection_lifecycle import lifecycle_class as geometry_class
from sciona.trackml_neighbors import CylinderNeighbors, CapNeighbors, Neighbors
from sciona.trackml_runtime import TrackRuntime
from sciona.trackml_execution import execute_tracking


def neighbor_class(cylinder_function, plane_function):
    if not callable(cylinder_function) or not callable(plane_function):
        raise ValueError('Two callable coordinate projection providers required')

    class Cylinder(CylinderNeighbors.Cylinder):
        def transformToNeighborCoords(self, points):
            return cylinder_function(points, self.cyl_mean_r2, self.zscale)

    class Cap(CapNeighbors.Cap):
        def transformToNeighborCoords(self, points):
            return plane_function(points, self.cap_z)

    class Cylinders(CylinderNeighbors):
        pass

    class Caps(CapNeighbors):
        pass

    Cylinders.Cylinder = Cylinder
    Caps.Cap = Cap

    class BoundNeighbors(Neighbors):
        pass

    BoundNeighbors.__init__ = _bound(Neighbors.__init__, dict(CylinderNeighbors=Cylinders, CapNeighbors=Caps))
    return BoundNeighbors


def lifecycle_class(circle_function, pitch_two_function, pitch_three_function, nearest_function,
                    cylinder_function, plane_function):
    neighbors = neighbor_class(cylinder_function, plane_function)
    base = geometry_class(circle_function, pitch_two_function, pitch_three_function, nearest_function)

    class Run(TrackRuntime.Run):
        pass

    Run.__init__ = _bound(TrackRuntime.Run.__init__, dict(Neighbors=neighbors))

    class Lifecycle(base):
        pass

    Lifecycle.Run = Run
    return Lifecycle


def execute(modules, hits, circle_function, pitch_two_function, pitch_three_function, nearest_function,
            cylinder_function, plane_function, *, cells=None, layer_functions=None, params=None, event_id=0):
    lifecycle = lifecycle_class(circle_function, pitch_two_function, pitch_three_function, nearest_function,
                                cylinder_function, plane_function)
    run = _bound(execute_tracking, dict(CheckedLifecycle=lifecycle))
    return run(modules, hits, cells=cells, layer_functions=layer_functions, params=params, event_id=event_id)

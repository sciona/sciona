"""Tracking with ten invocation-local numeric providers, including corrections."""
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_coordinate_lifecycle import lifecycle_class as coordinate_class
from sciona.decomposed_trackml_intersections import intersector_class
from sciona.decomposed_trackml_corrections import corrector_class
from sciona.trackml_execution import execute_tracking


def lifecycle_class(circle_function, pitch_two_function, pitch_three_function,
                    nearest_function, cylinder_function, plane_function,
                    cap_correction_function, cylinder_correction_function):
    base = coordinate_class(circle_function, pitch_two_function, pitch_three_function,
                            nearest_function, cylinder_function, plane_function)
    corrector = corrector_class(base.chooseLikelyNextHits.__globals__['HelixCorrector'],
                               cap_correction_function, cylinder_correction_function)
    class Lifecycle(base):
        pass
    Lifecycle.chooseLikelyNextHits = _bound(base.chooseLikelyNextHits, {'HelixCorrector': corrector})
    return Lifecycle


def execute(modules, hits, circle_function, pitch_two_function, pitch_three_function,
            nearest_function, cylinder_function, plane_function,
            cylinder_intersection_function, plane_intersection_function,
            cap_correction_function, cylinder_correction_function,
            *, cells=None, layer_functions=None, params=None, event_id=0):
    lifecycle = lifecycle_class(circle_function, pitch_two_function, pitch_three_function,
                                nearest_function, cylinder_function, plane_function,
                                cap_correction_function, cylinder_correction_function)
    intersector = intersector_class(cylinder_intersection_function, plane_intersection_function)
    run = _bound(execute_tracking, dict(CheckedLifecycle=lifecycle, Intersector=intersector))
    return run(modules, hits, cells=cells, layer_functions=layer_functions, params=params, event_id=event_id)

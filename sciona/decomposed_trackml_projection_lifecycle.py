"""Invocation-local nearest-point binding alongside circle and pitch providers."""
from functools import partial
from sciona.decomposed_trackml_lifecycle import lifecycle_class as geometry_class, _bound
from sciona.decomposed_trackml_nearest import nearest
from sciona.trackml_execution import execute_tracking
from sciona.trackml_extension import TrackExtension


def lifecycle_class(circle_function, pitch_two_function, pitch_three_function, nearest_function):
    if not callable(nearest_function):
        raise ValueError('Callable nearest-point provider required')
    base = geometry_class(circle_function, pitch_two_function, pitch_three_function)
    class Lifecycle(base):
        pass
    replacements = dict(helixNearestPointDistance=partial(nearest, operation=nearest_function))
    Lifecycle.findHelixIntersectionNeighbors = _bound(
        TrackExtension.findHelixIntersectionNeighbors, replacements)
    Lifecycle.evaluateLocally = _bound(base.evaluateLocally, replacements)
    return Lifecycle


def execute(modules, hits, circle_function, pitch_two_function, pitch_three_function,
            nearest_function, *, cells=None, layer_functions=None, params=None, event_id=0):
    lifecycle = lifecycle_class(circle_function, pitch_two_function, pitch_three_function, nearest_function)
    run = _bound(execute_tracking, dict(CheckedLifecycle=lifecycle))
    return run(modules, hits, cells=cells, layer_functions=layer_functions, params=params, event_id=event_id)

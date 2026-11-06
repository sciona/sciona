"""Invocation-local numeric bindings for retained TrackML lifecycle methods.

The retained methods use module-global numeric functions. Copy their function
namespaces for each invocation so explicit providers can replace those names
without mutating shared modules or duplicating the source algorithm. Corrector
clones and subsets retain the same invocation's bindings. Function code, default
arguments and all unrelated dependencies remain the retained implementations.
"""
from types import FunctionType
from sciona.trackml_execution import CheckedLifecycle,execute_tracking
from sciona.trackml_extension import TrackExtension
from sciona.trackml_fitting import FittedTrackExtension
from sciona.trackml_corrections import HelixCorrector
from sciona.decomposed_trackml_geometry import bind


def _bound(function,replacements):
    if function.__closure__:
        raise ValueError('Closure-bearing source methods require explicit review')
    namespace=dict(function.__globals__)
    if not set(replacements)<=set(namespace):
        raise ValueError('Expected retained dependency names are missing')
    namespace.update(replacements)
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__)
    result.__kwdefaults__=None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    result.__annotations__=dict(function.__annotations__)
    return result


def lifecycle_class(circle_function,pitch_two_function,pitch_three_function):
    operations=bind(circle_function,pitch_two_function,pitch_three_function)
    numeric=dict(circleFromThreePoints=operations['circle_operation'],
                 helixPitchFromTwoPoints=operations['pitch_two_operation'])
    class Corrector(HelixCorrector):pass
    Corrector.updateHelices=_bound(HelixCorrector.updateHelices,numeric)
    for name in ('clone','subset'):
        setattr(Corrector,name,_bound(getattr(HelixCorrector,name),{'HelixCorrector':Corrector}))
    class Lifecycle(CheckedLifecycle):pass
    Lifecycle.fitTracks=_bound(FittedTrackExtension.fitTracks,
        dict(numeric,helixPitchLeastSquares=operations['pitch_three_operation']))
    Lifecycle.getHelixParams=_bound(TrackExtension.getHelixParams,
        {'helixPitchFromTwoPoints':operations['pitch_two_operation']})
    Lifecycle.chooseLikelyNextHits=_bound(TrackExtension.chooseLikelyNextHits,{'HelixCorrector':Corrector})
    return Lifecycle


def execute(modules,hits,circle_function,pitch_two_function,pitch_three_function,*,
            cells=None,layer_functions=None,params=None,event_id=0):
    lifecycle=lifecycle_class(circle_function,pitch_two_function,pitch_three_function)
    run=_bound(execute_tracking,{'CheckedLifecycle':lifecycle})
    return run(modules,hits,cells=cells,layer_functions=layer_functions,params=params,event_id=event_id)

"""Bind generic clustering while preserving detector discovery recipe policy."""
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_commit_lifecycle import execute as commit_execute
from sciona.decomposed_trackml_bayesian_lifecycle import execute as bayesian_execute
from sciona.decomposed_trackml_neighbor_lifecycle import execute as neighbor_execute
from sciona.trackml_execution import execute_tracking
from sciona.trackml_detector import CapsSpec, DetectorSpec, discover_detector


def discovery_function(kmeans_function, mean_shift_function):
    if not all(callable(fn) for fn in (kmeans_function, mean_shift_function)):
        raise ValueError('Two callable clustering providers required')

    class KMeans:
        def __init__(self, *, n_clusters, n_init, random_state):
            self.options = dict(n_clusters=n_clusters, n_init=n_init, random_state=random_state)

        def fit(self, points):
            self.cluster_centers_, self.labels_ = kmeans_function(points, **self.options)
            return self

    class MeanShift:
        def __init__(self, *, bandwidth):
            self.bandwidth = bandwidth

        def fit(self, points):
            self.cluster_centers_, self.labels_ = mean_shift_function(
                points, bandwidth=self.bandwidth, n_jobs=1)
            return self

    class Caps(CapsSpec):
        pass

    Caps.__init__ = _bound(CapsSpec.__init__, dict(KMeans=KMeans, MeanShift=MeanShift))

    class Detector(DetectorSpec):
        pass

    Detector.__init__ = _bound(DetectorSpec.__init__, {'CapsSpec': Caps})
    return _bound(discover_detector, {'DetectorSpec': Detector})


def execute(*args, kmeans_function, mean_shift_function, **kwargs):
    discovery = discovery_function(kmeans_function, mean_shift_function)
    tracking = _bound(execute_tracking, {'discover_detector': discovery})
    neighbors = _bound(neighbor_execute, {'execute_tracking': tracking})
    bayesian = _bound(bayesian_execute, {'neighbor_execute': neighbors})
    return _bound(commit_execute, {'bayesian_execute': bayesian})(*args, **kwargs)

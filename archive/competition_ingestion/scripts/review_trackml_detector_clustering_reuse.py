"""Synthetic reuse assessment of existing clustering atoms for detector discovery."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from threadpoolctl import threadpool_limits

from sciona.atoms.ml.sklearn.cluster.atoms import mean_shift
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.trackml_detector import CapsSpec
from scripts.validate_trackml_detector import synthetic_modules

ROOT = Path(__file__).resolve().parents[1]


def review():
    calls = []

    class ExistingMeanShift:
        def __init__(self, *, bandwidth):
            self.bandwidth = bandwidth

        def fit(self, points):
            self.cluster_centers_, self.labels_ = mean_shift(
                points, bandwidth=self.bandwidth, n_jobs=1)
            calls.append(1)
            return self

    class ReusedCaps(CapsSpec):
        pass

    ReusedCaps.__init__ = _bound(CapsSpec.__init__, {'MeanShift': ExistingMeanShift})
    detector_cases = 0
    generic_cases = 0
    with threadpool_limits(limits=1):
        for scale in (.8, 1., 1.3):
            for paired in (False, True):
                for seed in (0, 1):
                    modules = synthetic_modules(scale, paired).sample(frac=1, random_state=seed)
                    before = modules.copy(deep=True)
                    geometry = SimpleNamespace(detectors_df=modules)
                    expected, actual = CapsSpec(geometry), ReusedCaps(geometry)
                    assert vars(expected).keys() == vars(actual).keys()
                    for name, value in vars(expected).items():
                        if isinstance(value, pd.DataFrame):
                            pd.testing.assert_frame_equal(value, getattr(actual, name), check_exact=True)
                        else:
                            np.testing.assert_array_equal(value, getattr(actual, name))
                    pd.testing.assert_frame_equal(modules, before, check_exact=True)
                    detector_cases += 1
        for seed in range(8):
            for dimensions in (1, 2, 4):
                rng = np.random.default_rng(seed)
                points = np.concatenate([rng.normal(loc=c, scale=.1, size=(12, dimensions))
                                         for c in (-3., 0., 3.)])
                before = points.copy()
                expected = MeanShift(bandwidth=.8, n_jobs=1).fit(points)
                centers, labels = mean_shift(points, bandwidth=.8, n_jobs=1)
                np.testing.assert_array_equal(centers, expected.cluster_centers_)
                np.testing.assert_array_equal(labels, expected.labels_)
                np.testing.assert_array_equal(points, before)
                generic_cases += 1
    import inspect
    paths = [Path(__file__), ROOT/'sciona/trackml_detector.py',
             ROOT/'scripts/validate_trackml_detector.py', Path(inspect.getfile(inspect.unwrap(mean_shift))),
             ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/sklearn/cluster/kmeans_public_api_shell/atoms.py']
    return dict(format='trackml-detector-clustering-reuse.v1', passed=True,
                approved=False, catalog_mutations=0, synthetic_only=True,
                exact_detector_cases=detector_cases, exact_generic_cases=generic_cases,
                injected_mean_shift_calls=len(calls), native_threads=1, workers=1,
                runtime_versions={name: metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'joblib')},
                selected_existing_provider=mean_shift.__module__+'.'+mean_shift.__name__,
                software_sha256={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths},
                limitations=[
                    'Float64 synthetic qualification only; existing atom coerces float32 to float64.',
                    'MeanShift centers and label order match the installed backend in these cases; no universal clustering accuracy claim.',
                    'Fixed ring count, radial gap policy, volume classification and layer reachability remain detector recipe policy.',
                    'KMeans fit execution remains unextracted; inspected public API shell only packages calls and fitted state.',
                    'This reuse assessment does not approve the provider or original CDG.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_detector_clustering_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('software_sha256', 'limitations')}))

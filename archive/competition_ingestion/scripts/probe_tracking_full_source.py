#!/usr/bin/env python3
"""Probe original tracking execution on mathematical fixtures, without approval.

Run in a fresh process: reviewed source modules occupy their original import
names only for this process. No third-party files or global APIs are modified.
"""
import argparse
from collections import Counter
import contextlib
import functools
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import traceback
import types

import numpy as np
import pandas as pd

from sciona.competition_dataframe_compat import compile_as_matrix_compat
from scripts.validate_tracking_detector_geometry import synthetic_geometry


def load_sources(root, source_dir, library_dir):
    for package in ('trackml', 'trackml_solution'):
        if package in sys.modules:
            raise RuntimeError('Run this probe in a fresh Python process')
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    manifests = [(library_dir, 'tracking_library_pins.json',
                  ['trackml/__init__.py', 'trackml/score.py', 'trackml/dataset.py']),
                 (source_dir, 'tracking_source_pins.json',
                  ['trackml_solution/'+n+'.py' for n in
                   ['logging', 'geometry', 'neighbors', 'candidates', 'corrections', 'cells', 'data', 'algorithm']])]
    for directory, manifest, files in manifests:
        pins = json.loads((root/'docs/reviews'/manifest).read_text())
        for filename in files:
            content = (directory/filename).read_bytes()
            if hashlib.sha256(content).hexdigest() != pins['source_files'][filename]:
                raise ValueError('Source pin mismatch: '+filename)
            name = filename[:-3].replace('/', '.').removesuffix('.__init__')
            module = sys.modules.get(name) or types.ModuleType(name)
            module.__file__ = str(directory/filename)
            sys.modules[name] = module
            if name.endswith('.neighbors') or name.endswith('.cells'):
                code, helpers = compile_as_matrix_compat(content,
                    expected_calls=2 if name.endswith('.neighbors') else 1, filename=filename)
                module.__dict__.update(helpers)
            else:
                code = compile(content, filename, 'exec')
            exec(code, module.__dict__)
    return sys.modules['trackml_solution.geometry'], sys.modules['trackml_solution.algorithm'], sys.modules['trackml_solution.data']


def probe(root, source_dir, library_dir, *, commit_limit=1000):
    calls, completed = Counter(), Counter()
    result = dict(approved=False, synthetic_only=True, full_execution_passed=False,
                  scope='Original source execution probe; no model-quality or catalog approval claim.',
                  compatibility={'as_matrix_calls': 3}, entered=calls, completed=completed,
                  excluded_paths=['truth scoring', 'cell features', 'learned layer calibration',
                                  'nonphysical postprocessing', 'hyperparameter optimization'],
                  runtime_versions={name: importlib.metadata.version(name) for name in
                                    ['numpy', 'pandas', 'scipy', 'scikit-learn']},
                  evidence_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                                   ['docs/reviews/tracking_source_pins.json',
                                    'docs/reviews/tracking_library_pins.json',
                                    'sciona/competition_dataframe_compat.py',
                                    'scripts/validate_tracking_detector_geometry.py']})
    try:
        geometry_source, algorithm_source, data_source = load_sources(root, source_dir, library_dir)
        geometry = synthetic_geometry(geometry_source)
        detector = geometry_source.DetectorSpec(geometry)
        # Synthetic layer occupancy, plus exact helical observations through the
        # three generated cylinders. No detector/event records are copied.
        hits = geometry.detectors_df[['volume_id', 'layer_id', 'cx', 'cy', 'cz']].rename(
            columns={'cx': 'x', 'cy': 'y', 'cz': 'z'}).copy()
        extra = []
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            for layer, radius in enumerate([20., 40., 60.]):
                phase = 2*np.arcsin(radius/400.)
                x, y = 200*np.sin(phase), 200*(1-np.cos(phase))
                extra.append(dict(volume_id=geometry_source.CylindersSpec.cylinder_volume_ids[0],
                    layer_id=layer, x=x*np.cos(angle)-y*np.sin(angle),
                    y=x*np.sin(angle)+y*np.cos(angle), z=10*phase))
        hits = pd.concat([hits, pd.DataFrame(extra)], ignore_index=True)
        hits['hit_id'] = np.arange(1, len(hits)+1)
        hits['module_id'] = 1
        methods = ['setupRun', 'chooseLikelyFirstHits', 'chooseLikelySecondHits',
                   'chooseLikelyNextHits', 'findPairs', 'fitTracks', 'evaluateTracks',
                   'dropRedundantTracks', 'filterInvalidTrackCandidates']
        class ObservedAlgorithm(algorithm_source.Algorithm):
            pass
        for name in methods:
            original = getattr(algorithm_source.Algorithm, name)
            def wrap(fn, method):
                @functools.wraps(fn)
                def observed(self, *args, **kwargs):
                    calls[method] += 1
                    value = fn(self, *args, **kwargs)
                    completed[method] += 1
                    return value
                return observed
            setattr(ObservedAlgorithm, name, wrap(original, name))
        # The source mutates class defaults. Copy both default dictionaries in
        # this disposable process before constructing the algorithm.
        ObservedAlgorithm.default_params = algorithm_source.Algorithm.default_params.copy()
        algorithm_source.Neighbors.default_params = algorithm_source.Neighbors.default_params.copy()
        algo = ObservedAlgorithm(detector, max_log_indent=-1, params={
            'follow__niter': 3, 'rank__ntop_qu': 1., 'rank__ntop': 1000,
            'follow__drop_start': 0, 'commit__nmax': commit_limit,
            'commit__niter': 3})
        with tempfile.TemporaryDirectory(prefix='tracking-synthetic-') as temp:
            prefix = Path(temp)/'event000001'
            hits.to_csv(str(prefix)+'-hits.csv', index=False)
            event = data_source.Event(str(prefix), with_truth=False, with_cells=False)
            output = Path(temp)/'result.csv'
            algo.findTracks(None, [event], submission_filename=str(output),
                            analysis=False, score_intermediate=False, score_final=False)
            submission = pd.read_csv(output)
            np.testing.assert_array_equal(np.sort(submission.hit_id), hits.hit_id)
            tracks = submission.set_index('hit_id').track_id
            recovered = 0
            recovered_track_ids = []
            for offset in range(len(hits)-24+1, len(hits)+1, 3):
                identities = tracks.loc[list(range(offset, offset+3))].to_numpy()
                recovered += int(identities[0] != 0 and np.all(identities == identities[0]))
                recovered_track_ids.append(int(identities[0]))
            if recovered != 8:
                raise AssertionError('Synthetic helix recovery failed')
            if len(set(recovered_track_ids)) != 8:
                raise AssertionError('Distinct synthetic helices were merged')
            if any((tracks == identity).sum() != 3 for identity in recovered_track_ids):
                raise AssertionError('Recovered synthetic helix includes unrelated observations')
            if any(completed[method] == 0 for method in methods) or calls != completed:
                raise AssertionError('Required algorithm stage did not complete')
            if commit_limit == 1 and completed['setupRun'] < 2:
                raise AssertionError('Expected multiple commitment rounds')
            result.update(full_execution_passed=True, observation_count=len(hits),
                          assigned_observations=int((submission.track_id != 0).sum()),
                          synthetic_helices_recovered=recovered, synthetic_helices_total=8,
                          helix_membership_exact=True,
                          commit_limit=commit_limit)
    except Exception as error:
        result['failure_type'] = type(error).__name__
        result['failure'] = str(error)
        result['traceback'] = traceback.format_exc()
    result['validator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit-limit', type=int, default=1000)
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        result = probe(Path(__file__).resolve().parents[1], args.source_dir, args.library_dir,
                       commit_limit=args.commit_limit)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

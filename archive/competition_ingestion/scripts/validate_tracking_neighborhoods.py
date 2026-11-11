#!/usr/bin/env python3
"""Validate source neighborhoods and commitment on generated geometry only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from sciona.competition_dataframe_compat import compile_as_matrix_compat
from scripts.validate_tracking_detector_geometry import synthetic_geometry


def validate(root, source_dir):
    pins = json.loads((root/'docs/reviews/tracking_source_pins.json').read_text())

    def load(name, conversions=0):
        filename = 'trackml_solution/'+name+'.py'
        path = source_dir/filename
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != pins['source_files'][filename]:
            raise ValueError('Source pin mismatch')
        spec = importlib.util.spec_from_file_location('reviewed_'+name, path)
        module = importlib.util.module_from_spec(spec)
        if conversions:
            code, helpers = compile_as_matrix_compat(content, expected_calls=conversions, filename=str(path))
            module.__dict__.update(helpers)
            exec(code, module.__dict__)
        else:
            spec.loader.exec_module(module)
        return module

    source = load('geometry')
    neighbor_source = load('neighbors', conversions=2)
    candidate_source = load('candidates')
    geometry = synthetic_geometry(source)
    detector = source.DetectorSpec(geometry)
    observations = geometry.detectors_df[['volume_id', 'layer_id', 'cx', 'cy', 'cz']].rename(
        columns={'cx': 'x', 'cy': 'y', 'cz': 'z'}).copy()
    observations['hit_id'] = np.arange(1, len(observations)+1)
    in_cylinder = np.zeros(len(observations)+1, dtype=bool)
    layer_ids = np.full(len(observations)+1, -1, dtype=np.int8)
    neighbors = neighbor_source.Neighbors(detector)
    neighbors.fit(observations, hit_in_cyl=in_cylinder, hit_layer_id=layer_ids)
    if np.any(layer_ids[1:] < 0):
        raise ValueError('Generated observations were not assigned to layers')
    queries = 0
    for index, observation in observations.iterrows():
        identity = int(observation['hit_id'])
        point = observation[['x', 'y', 'z']].to_numpy(dtype=float)[None, :]
        finder = neighbors.cyln if in_cylinder[identity] else neighbors.capn
        result = finder.findNeighborhoodK(int(layer_ids[identity]), point, 1)
        if len(result) != 1 or int(result['nb_hit_id'].iloc[0]) != identity:
            raise ValueError('Nearest-neighbor identity differs for exact synthetic point')
        if not np.isclose(float(result['nb_dist'].iloc[0]), 0., atol=1e-12):
            raise ValueError('Exact self-neighbor has nonzero distance')
        queries += 1
    # Sequential commitment must remove already claimed observations without
    # consuming new observations from a rejected candidate.
    candidates = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]], dtype=np.int64)
    used = np.zeros(len(observations)+1, dtype=bool)
    candidate_source.zeroUsedHits(candidates, used, min_nhits=3)
    np.testing.assert_array_equal(candidates, [[1, 2, 3], [0, 0, 0], [4, 5, 6]])
    np.testing.assert_array_equal(np.flatnonzero(used), np.arange(7))
    remaining = observations.loc[~used[observations['hit_id'].to_numpy()]]
    refreshed = neighbor_source.Neighbors(detector)
    refreshed.fit(remaining)
    # Querying a previously consumed point must no longer return a used ID.
    first = observations.iloc[0]
    result = refreshed.cyln.findNeighborhoodK(0, first[['x', 'y', 'z']].to_numpy(dtype=float)[None, :], 1)
    if used[int(result['nb_hit_id'].iloc[0])]:
        raise ValueError('Rebuilt neighborhood includes committed observation')
    files = ['scripts/validate_tracking_neighborhoods.py', 'scripts/validate_tracking_detector_geometry.py',
             'sciona/competition_dataframe_compat.py', 'tests/test_competition_dataframe_compat.py']
    return dict(approved=False, synthetic_only=True, source_commit=pins['commit'],
        reviewed_as_matrix_rewrites=2, exact_self_neighbor_queries=queries,
        checks=dict(layer_assignment=True, nearest_neighbors=True, sequential_commitment=True, rebuilt_neighborhood_excludes_used=True),
        implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        scope='Synthetic detector -> original neighborhoods -> original commitment -> rebuilt neighborhoods. No complete seed/extension/ranking loop or publication approval.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.source_dir)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['reviewed_as_matrix_rewrites', 'exact_self_neighbor_queries', 'checks']}))

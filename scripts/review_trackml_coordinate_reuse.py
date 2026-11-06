"""Measure source compatibility of the existing cylindrical projection provider."""
import ast
import inspect
import json
from types import SimpleNamespace
import numpy as np
from sciona.atoms.particle_tracking.detector_corrections.atoms import coordinate_rescaling_for_knn
from sciona.trackml_neighbors import CylinderNeighbors, CapNeighbors
from sciona.atoms.physics.coordinate_projection import cylindrical_projection, plane_projection
from scripts.audit_decomposed_trackml_readiness import ROOT, sha


def review():
    pins = json.loads((ROOT/'docs/reviews/competition_trackml_source_pins.json').read_text())
    source = ROOT.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking/solution_trackml_solution_neighbors.py.source'
    if sha(source) != pins['files']['trackml_solution/neighbors.py']:
        raise ValueError('Pinned neighbor source changed')
    tree = ast.parse(source.read_text())
    methods = {}
    for outer, inner, key in [('CylinderNeighbors', 'Cylinder', 'cylinder'), ('CapNeighbors', 'Cap', 'cap')]:
        node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == outer]
        node, = [n for n in node.body if isinstance(n, ast.ClassDef) and n.name == inner]
        node, = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'transformToNeighborCoords']
        ns = dict(np=np)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-coordinate-transform>', 'exec'), ns)
        methods[key] = ns['transformToNeighborCoords']
    paths = ['scripts/review_trackml_coordinate_reuse.py', 'sciona/trackml_neighbors.py',
             'docs/reviews/competition_trackml_source_pins.json']
    before = {p: sha(ROOT/p) for p in paths}
    provider_path = inspect.getfile(inspect.unwrap(coordinate_rescaling_for_knn))
    provider_hash = sha(provider_path)
    projection_path = inspect.getfile(inspect.unwrap(cylindrical_projection))
    projection_hash = sha(projection_path)
    exact = different = elements = cap_cases = 0
    max_error = 0.
    for seed in range(12):
        rng = np.random.RandomState(seed + 1701)
        points = rng.normal(size=(64, 3))
        for radius in (.1, 1., 37.):
            for weight in (.2, 1., 5.):
                state = SimpleNamespace(cyl_mean_r2=radius, zscale=weight)
                saved = points.copy()
                expected = methods['cylinder'](state, points)
                retained = CylinderNeighbors.Cylinder.transformToNeighborCoords(state, points)
                np.testing.assert_array_equal(expected, retained)
                np.testing.assert_array_equal(cylindrical_projection(points, radius, weight), expected)
                actual = coordinate_rescaling_for_knn(points, cyl_mean_r2=radius, z_scale=weight)
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
                mismatch = actual != expected
                if mismatch.any():
                    different += 1
                else:
                    exact += 1
                elements += int(mismatch.sum())
                max_error = max(max_error, float(np.max(np.abs(actual-expected))))
                np.testing.assert_array_equal(points, saved)
        for height in (-12., .5, 18.):
            state = SimpleNamespace(cap_z=height)
            expected = methods['cap'](state, points)
            np.testing.assert_array_equal(CapNeighbors.Cap.transformToNeighborCoords(state, points), expected)
            np.testing.assert_array_equal(plane_projection(points, height), expected)
            np.testing.assert_allclose(expected, points[:, :2]*(height/points[:, 2])[:, None], rtol=0, atol=0)
            cap_cases += 1
    origin = np.array([[0., 0., 1.]])
    with np.errstate(divide='ignore', invalid='ignore'):
        source_origin = methods['cylinder'](SimpleNamespace(cyl_mean_r2=2., zscale=1.), origin)
    provider_origin = coordinate_rescaling_for_knn(origin, cyl_mean_r2=2.)
    if np.isfinite(source_origin).all() or not np.isfinite(provider_origin).all():
        raise AssertionError('Expected zero-radius semantic difference')
    if not different:
        raise AssertionError('Expected arithmetic-order differences')
    if before != {p: sha(ROOT/p) for p in paths} or sha(provider_path) != provider_hash or sha(projection_path) != projection_hash:
        raise ValueError('Reviewed implementation changed')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        cylinder_cases=exact+different, exact_existing_provider_cases=exact,
        rounded_difference_cases=different, differing_elements=elements,
        maximum_absolute_difference=max_error, exact_cap_source_cases=cap_cases,
        exact_new_cylinder_provider_cases=exact+different, exact_new_plane_provider_cases=cap_cases,
        projection_provider_sha256=projection_hash,
        source_sha256=sha(source), provider_sha256=provider_hash, implementation_sha256=before,
        zero_radius_difference='Source produces nonfinite geometry; existing provider substitutes a finite result.',
        decision='Do not substitute this provider under an exact source-parity claim. Expose generic cylindrical and plane projections preserving the source arithmetic order, with caller-provided radius, plane height and transverse weight.',
        limits=['Small coordinate differences can affect threshold or tie decisions; this review does not quantify downstream changes.',
                'No real detector data or runtime records are persisted.',
                'New providers are locally registered; lifecycle integration and canonical qualification remain pending.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_coordinate_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

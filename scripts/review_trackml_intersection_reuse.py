"""Compare reusable intersection atoms against pinned source using synthetic geometry."""
import ast
import inspect
import json
from types import SimpleNamespace
import numpy as np
from sciona.atoms.particle_tracking.track_matching.atoms import helix_cap_intersection, helix_cylinder_intersection
from sciona.trackml_intersections import CapIntersector, CylinderIntersector
from scripts.audit_decomposed_trackml_readiness import ROOT, sha


def source_classes():
    pins = json.loads((ROOT/'docs/reviews/competition_trackml_source_pins.json').read_text())
    source = ROOT.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking/solution_trackml_solution_geometry.py.source'
    if sha(source) != pins['files']['trackml_solution/geometry.py']:
        raise ValueError('Pinned geometry source changed')
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in ('CapIntersector', 'CylinderIntersector')]
    if len(nodes) != 2:
        raise ValueError('Expected two pinned intersection classes')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-intersections>', 'exec'), namespace)
    return namespace, source


def review():
    source, source_path = source_classes()
    paths = {'reviewer': __file__, 'retained': ROOT/'sciona/trackml_intersections.py',
             'provider': inspect.getfile(inspect.unwrap(helix_cap_intersection)), 'source': source_path}
    before = {k: sha(p) for k, p in paths.items()}
    cap_cases = 0
    for seed in range(12):
        rng = np.random.RandomState(seed+831)
        for height in (-17., -.5, .5, 17.):
            n = 32
            xm, ym = rng.normal(size=(2, n))
            radius = rng.uniform(.2, 4., n)
            phase = rng.uniform(-np.pi, np.pi, n)
            x, y = xm+radius*np.cos(phase), ym+radius*np.sin(phase)
            z = np.zeros(n)
            pitch = rng.uniform(.3, 7., n)*rng.choice([-1., 1.], n)
            direction = np.full(n, np.sign(height))
            spec = SimpleNamespace(cap_z=np.array([height]), ring_overlap_r2_min_sqr=[], ring_overlap_r2_max_sqr=[])
            args = (x, y, z, direction, xm, ym, radius, pitch)
            saved = [a.copy() for a in args]
            expected = source['CapIntersector'](spec).intersectHelices(*args, pre_move=0., missable=False)
            retained = CapIntersector(spec).intersectHelices(*args, pre_move=0., missable=False)
            actual = helix_cap_intersection(x, y, z, xm, ym, radius, pitch, np.full(n, height))
            for a, b in zip(retained, expected, strict=True):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(actual, (expected[0], expected[1], expected[2], expected[4]), strict=True):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(expected[3], np.zeros(n, dtype=np.int8))
            np.testing.assert_allclose((actual[0]-xm)**2+(actual[1]-ym)**2, radius**2, rtol=2e-13, atol=2e-13)
            for a, b in zip(args, saved, strict=True):
                np.testing.assert_array_equal(a, b)
            cap_cases += 1
    # One helix: center (2,0), radius 1, phase .5, positive axial motion.
    x, y, z, xm, ym, radius, pitch, direction = [np.array([v]) for v in
        (2.+np.cos(.5), np.sin(.5), 0., 2., 0., 1., 2.*np.pi, 1.)]
    args = (x, y, z, direction, xm, ym, radius, pitch)
    differences = []
    for name, target, move in [('minimum_movement', 5.+4.*np.cos(.6), .2), ('unreachable_target', 100., 0.)]:
        spec = SimpleNamespace(cyl_rsqr=np.array([target]))
        expected = source['CylinderIntersector'](spec).intersectHelices(*args, pre_move=np.array([move]), missable=False)
        retained = CylinderIntersector(spec).intersectHelices(*args, pre_move=np.array([move]), missable=False)
        actual = helix_cylinder_intersection(x, y, z, xm, ym, radius, pitch, direction, np.array([target]))
        for a, b in zip(retained, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        if name == 'minimum_movement':
            if not (actual[3][0] < move < expected[4][0]):
                raise AssertionError('Expected minimum-movement branch difference')
        else:
            if expected[3][0] != -1 or not np.isnan(expected[0][0]) or not all(np.isfinite(a).all() for a in actual):
                raise AssertionError('Expected unreachable-target miss difference')
            if np.isclose(actual[0][0]**2+actual[1][0]**2, target):
                raise AssertionError('Clipped result unexpectedly lies on target')
        differences.append(name)
    if before != {k: sha(p) for k, p in paths.items()}:
        raise ValueError('Reviewed implementation changed')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        exact_plane_batches=cap_cases, helices_per_plane_batch=32,
        cylinder_incompatibilities= differences, implementation_sha256=before,
        decision=dict(plane='Reuse existing plane-intersection atom for nonempty finite aligned arrays with nonzero pitch; retain target selection, miss handling and correction hooks in the recipe.',
                      cylinder='Existing atom is not source-compatible for minimum movement or unreachable targets. Expose caller-controlled phase tolerance and explicit miss semantics in a reusable numerical provider.'),
        limitations=['Synthetic comparisons do not qualify original CDG publication or historical accuracy.',
                     'Empty selected batches need an explicit adapter because the plane atom requires at least one helix.',
                     'Plane radius argument is unused by the coordinate calculation; radius remains needed for recipe arc length.',
                     'Detector layer selection, lateral misses, recursive retries, correction and multiplicity remain recipe responsibilities.',
                     'No catalog promotion is performed by this review.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_intersection_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

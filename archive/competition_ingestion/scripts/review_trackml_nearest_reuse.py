"""Read-only qualification of existing nearest-point geometry for TrackML reuse."""
import ast
import inspect
import json
from pathlib import Path

import icontract
import numpy as np
from sciona.atoms.particle_tracking.helix_geometry import atoms
from sciona.decomposed_trackml_nearest import nearest
from sciona.trackml_nearest import nearest_point_distance
from scripts.audit_decomposed_trackml_readiness import ROOT, sha, connect


def review():
    provider = atoms.helix_nearest_point_distance
    provider_path = Path(inspect.getfile(inspect.unwrap(provider)))
    provider_hash = sha(provider_path)
    if provider_hash != '26677db3581b8be023b489ddf956d71328aea5e66631afa4c00b3601914dfd42':
        raise ValueError('Reviewed provider source changed')
    paths = ['scripts/review_trackml_nearest_reuse.py', 'sciona/decomposed_trackml_nearest.py',
             'sciona/trackml_nearest.py', 'sciona/trackml_helix.py',
             'sciona/trackml_extension.py', 'sciona/trackml_ranking.py',
             'tests/test_decomposed_trackml_nearest.py',
             'docs/reviews/competition_trackml_source_pins.json']
    before = {p: sha(ROOT / p) for p in paths}
    pins = json.loads((ROOT / paths[-1]).read_text())
    vendor = provider_path.parent.parent / 'source_tracking'
    source_hashes = {}
    trees = {}
    for name in ('geometry', 'algorithm'):
        path = vendor / f'solution_trackml_solution_{name}.py.source'
        digest = sha(path)
        if digest != pins['files'][f'trackml_solution/{name}.py']:
            raise ValueError('Pinned source mismatch')
        source_hashes[name] = digest
        trees[name] = ast.parse(path.read_text())
    nodes = [n for n in trees['geometry'].body
             if isinstance(n, ast.FunctionDef) and n.name == 'helixNearestPointDistance']
    if len(nodes) != 1:
        raise ValueError('Expected one source function')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-nearest>', 'exec'), namespace)
    reference = namespace['helixNearestPointDistance']
    calls = {}
    for name, tree in [('source_algorithm', trees['algorithm']),
                       *[(p, ast.parse((ROOT / p).read_text())) for p in paths[4:6]]]:
        found = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'helixNearestPointDistance']
        # These exact source call sites have no optional-output or iteration override.
        if not found or any(n.keywords or len(n.args) != 4 or
                            not isinstance(n.args[0], ast.Starred) for n in found):
            raise ValueError('Nearest-point recipe call contract changed')
        calls[name] = len(found)
    if calls['source_algorithm'] != 2 or sum(calls[p] for p in paths[4:6]) != 2:
        raise ValueError('Nearest-point source call-site coverage changed')
    comparisons = analytic = 0
    for seed in range(12):
        rng = np.random.RandomState(seed + 421)
        n = 32
        xm, ym, z0 = rng.normal(size=(3, n))
        radius = rng.uniform(.5, 5, n)
        pitch = rng.uniform(1, 20, n) * rng.choice([-1., 1.], n)
        phi = rng.uniform(-np.pi, np.pi, n)
        params = [xm + radius * np.cos(phi), ym + radius * np.sin(phi), z0,
                  xm, ym, radius, pitch]
        targets = list(rng.normal(size=(3, n)) * 5)
        for scale in (.01, 1., 100.):
            for scalar in (False, True):
                values = [float(v[0]) * scale if scalar else v * scale for v in params]
                values += [v * scale for v in targets]
                for iterations in (1, 3, 6):
                    saved = [np.asarray(v).copy() for v in values]
                    expected = reference(*values, iterations=iterations)
                    if not all(np.isfinite(v).all() for v in expected):
                        raise ValueError('Comparison must have finite source outputs')
                    actual = provider(*values, iterations=iterations)
                    adapted = nearest(*values, iterations=iterations, operation=provider)
                    for a, b, c in zip(actual, adapted, expected, strict=True):
                        np.testing.assert_array_equal(a, c)
                        np.testing.assert_array_equal(b, c)
                    for a, b in zip(values, saved, strict=True):
                        np.testing.assert_array_equal(a, b)
                    comparisons += 1
            # Independent known solution: target lies on the helix, across multiple turns.
            delta = rng.uniform(-4 * np.pi, 4 * np.pi, n)
            target = [xm + radius * np.cos(phi + delta), ym + radius * np.sin(phi + delta),
                      z0 + pitch * delta / (2 * np.pi)]
            output = nearest(*[v * scale for v in params + target], operation=provider)
            for actual, expected in zip(output[:3], target, strict=True):
                np.testing.assert_allclose(actual, expected * scale, atol=1e-11 * scale, rtol=1e-12)
            np.testing.assert_allclose(output[3], 0, atol=1e-11 * scale)
            analytic += 1
    empty = [np.empty(0) for _ in range(10)]
    try:
        provider(*empty)
    except icontract.ViolationError:
        pass
    else:
        raise AssertionError('Expected provider nonempty precondition')
    for a, b in zip(nearest(*empty, operation=provider), reference(*empty), strict=True):
        np.testing.assert_array_equal(a, b)
    # Zero pitch is undefined. Compare rejection, not an invented finite substitute.
    undefined = [1., 0., 0., 0., 0., 1., 0., np.ones(1), np.zeros(1), np.zeros(1)]
    for function in (nearest_point_distance, lambda *a: nearest(*a, operation=provider)):
        try:
            function(*undefined)
        except (ValueError, icontract.ViolationError):
            pass
        else:
            raise AssertionError('Undefined zero-pitch projection accepted')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows = db.execute('''SELECT a.fqdn,a.status,a.is_publishable,a.import_module,a.source_symbol,
            v.version_id::text,v.content_hash,v.trust_tier,
            EXISTS(SELECT 1 FROM catalog_atoms_served s WHERE s.atom_id=a.atom_id) AS served
            FROM atoms a JOIN atom_versions v USING(atom_id)
            WHERE v.is_latest AND a.import_module=%s AND a.source_symbol=%s''',
            (atoms.__name__, provider.__name__)).fetchall()
    if len(rows) != 1:
        raise ValueError('Expected exactly one catalog candidate')
    if before != {p: sha(ROOT / p) for p in paths} or sha(provider_path) != provider_hash:
        raise ValueError('Compared implementation changed')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
                exact_source_comparisons=comparisons, analytic_on_helix_cases=analytic,
                checked_recipe_calls=calls, empty_batch_compatibility=True,
                zero_pitch_rejections=2, provider_sha256=provider_hash,
                pinned_source_sha256=source_hashes, implementation_sha256=before,
                catalog_candidate=rows[0],
                decision='Reuse the existing four-output provider through an explicit boundary; lifecycle integration and canonical qualification remain required.',
                limits=['Fixed-iteration approximation, not a guaranteed global nearest point.',
                        'Optional phase and arc outputs are outside the tracking call-site scope reviewed here.',
                        'Finite synthetic comparisons do not establish equivalence for arbitrary geometry.',
                        'No original CDG or new canonical provider approval is established by this report.'])


if __name__ == '__main__':
    result = review()
    (ROOT / 'docs/reviews/trackml_nearest_reuse.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

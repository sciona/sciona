"""Check an existing numeric decision atom against pinned public source."""
import ast
import inspect
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sciona.atoms.particle_tracking.track_matching.atoms import bayesian_neighbor_evaluation
from sciona.trackml_bayesian import BayesianNeighborScorer
from scripts.audit_decomposed_trackml_readiness import ROOT, sha


def source_method():
    pins = json.loads((ROOT/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path = ROOT.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking/solution_trackml_solution_algorithm.py.source'
    if sha(path) != pins['files']['trackml_solution/algorithm.py']:
        raise ValueError('Pinned algorithm changed')
    methods = [n for n in ast.walk(ast.parse(path.read_text()))
               if isinstance(n, ast.FunctionDef) and n.name == 'bayesianNeighborEvaluation']
    if len(methods) != 1: raise ValueError('Unique source method required')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<pinned-bayesian>', 'exec'), namespace)
    return namespace['bayesianNeighborEvaluation'], path


def compare(source, dtheta, dphi, btheta, bphi, etheta, ephi, cut, trust):
    n = len(dtheta)
    zero = np.zeros(n)
    # Synthetic API inputs only. Geometry projection and error prediction are
    # supplied independently so this review isolates the numeric decision.
    scorer = SimpleNamespace(projectNeighborDisplacement=lambda *a: (dtheta, dphi),
                             predictRandomError=lambda *a, **k: (etheta, ephi))
    run = SimpleNamespace(event=SimpleNamespace(hitCoordinatesById=lambda ids: (zero, zero, zero)),
        neighbors=SimpleNamespace(evaluateLayerFunctions=lambda *a, **k: (btheta, bphi, etheta, ephi)),
        params={'nb__cut_factor': cut, 'nb__dist_trust': trust})
    frame = pd.DataFrame(dict(extend_hit_id=np.arange(n), extend_index=np.arange(n),
                             xn=zero, yn=zero, zn=zero, cyl_closer=np.zeros(n, dtype=bool),
                             next_id=np.zeros(n, dtype=int), hel_s=zero))
    details = SimpleNamespace()
    expected = source(scorer, run, frame, details=details)
    retained = BayesianNeighborScorer.bayesianNeighborEvaluation(scorer, run, frame)
    actual = bayesian_neighbor_evaluation(dtheta, dphi, btheta, bphi, etheta, ephi,
                                          cut_factor=cut, dist_trust=trust)
    for a, b, c in zip(actual, expected, retained, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(b, c)
    return actual, details


def review():
    source, path = source_method()
    paths = dict(source=path, retained=ROOT/'sciona/trackml_bayesian.py', reviewer=__file__,
                 provider=inspect.getfile(inspect.unwrap(bayesian_neighbor_evaluation)))
    before = {k: sha(p) for k, p in paths.items()}
    batches = 0
    for seed in range(12):
        rng = np.random.RandomState(2841+seed)
        for scale in (.01, 1., 100.):
            residual = rng.normal(size=(2, 32))*scale
            background = rng.uniform(0., 3., (2, 32))*scale
            error = rng.uniform(.1, 2., (2, 32))*scale
            for cut, trust in ((1., .8), (3., .5)):
                compare(source, *residual, *background, *error, cut, trust)
                batches += 1
    one, zero = np.ones(1), np.zeros(1)
    _, details = compare(source, zero, zero, one, one, one, one, 1., .8)
    threshold = details.bay_cut
    equal, _ = compare(source, threshold, zero, one, one, one, one, 1., 1.)
    below, _ = compare(source, np.nextafter(threshold, -np.inf), zero, one, one, one, one, 1., 1.)
    above, _ = compare(source, np.nextafter(threshold, np.inf), zero, one, one, one, one, 1., 1.)
    if equal[0][0] or equal[1][0] or not below[0][0] or below[1][0] or above[0][0] or not above[1][0]:
        raise ValueError('Strict threshold boundary differs')
    at_zero, _ = compare(source, zero, zero, zero, zero, one, one, 1., .8)
    if at_zero[0][0] or at_zero[1][0]: raise ValueError('Zero-background strict cut differs')
    if before != {k: sha(p) for k, p in paths.items()}: raise ValueError('Reviewed software changed')
    return dict(format='trackml-bayesian-reuse.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, exact_source_batches=batches, batch_size=32,
        strict_boundary_cases=4, implementation_sha256=before,
        decision='Reuse the existing concrete decision atom with explicit recipe thresholds and qualified finite aligned inputs. Keep coordinate projection, error prediction, coefficient lookup and diagnostics in the recipe or separately reusable providers.',
        limitations=[
            'Residuals, background scales and errors must use matching units for each component; normalized ratios and both thresholds are dimensionless. Source projected residuals have length units despite angular terminology in existing metadata.',
            'Existing symbolic metadata uses angle dimensions and describes normalized distance only; this review does not qualify a full symbolic decision proof.',
            'Source recipe defaults are cut_factor=1.0 and dist_trust=0.8; provider defaults are 3.0 and 0.5. Always pass the recipe controls explicitly.',
            'Qualified domain is nonempty aligned finite float64 vectors, positive error scales and positive scalar thresholds, with finite intermediate arithmetic. Zero background is allowed.',
            'Good and dubious flags are independent strict comparisons; a point can be both good and dubious. Equality to either respective threshold does not pass that comparison.',
            'Source diagnostics include normalized distance and cut; the existing atom returns flags only. Integration must preserve diagnostics separately.',
            'This isolates decision arithmetic. Projection, empirical error model, lifecycle integration and catalog qualification remain separate work.',
            'Synthetic source parity does not establish calibrated probabilities, arbitrary-input accuracy or historical competition performance.',
        ])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_bayesian_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

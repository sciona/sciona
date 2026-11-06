"""Synthetic boundary assessment before extracting remaining reusable operations."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sciona.trackml_bayesian import BayesianNeighborScorer
from sciona.trackml_layer_functions import createLayerFunctionInterpolators

ROOT = Path(__file__).resolve().parents[1]


def review():
    scorer = BayesianNeighborScorer()
    projection_cases = error_cases = interpolation_cases = 0
    with threadpool_limits(limits=1):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            for sign in (-1, 0, 1):
                points = rng.uniform(.5, 5., (32, 3))
                points[:, 2] *= sign
                delta = rng.normal(size=points.shape)
                x, y, z = points.T
                radius = np.linalg.norm(points, axis=1)
                rho = np.linalg.norm(points[:, :2], axis=1)
                # Independent spherical-coordinate basis, with the source's
                # hemisphere orientation and degenerate equatorial convention.
                theta = np.arccos(z / radius)
                phi = np.arctan2(y, x)
                polar = np.column_stack((np.cos(theta)*np.cos(phi),
                                         np.cos(theta)*np.sin(phi), -np.sin(theta)))
                azimuth = np.column_stack((-y/rho, x/rho, np.zeros(len(x))))
                actual = scorer.projectNeighborDisplacement(*points.T, *(points-delta).T)
                np.testing.assert_allclose(actual[0], sign*np.sum(polar*delta, axis=1), atol=3e-14)
                np.testing.assert_allclose(actual[1], np.sum(azimuth*delta, axis=1), atol=3e-14)
                if sign == 0:
                    np.testing.assert_array_equal(actual[0], np.zeros(len(x)))
                projection_cases += 1
            for sign in (-1, 0, 1):
                distance = sign*rng.uniform(.1, 100., 32)
                measured = rng.uniform(.01, 2., (2, 32))
                actual = scorer.predictRandomError(None, distance, *measured)
                expected = [np.hypot(m, distance*.007884*c)
                            for m, c in zip(measured, (.2096, .3853))]
                np.testing.assert_allclose(actual, expected, rtol=3e-15)
                error_cases += 1
        for dimensions in (1, 2, 3):
            for descending in (False, True):
                for dropped in (0, 1, 2):
                    axes = [np.linspace(-2., 3., 4+i) for i in range(dimensions)]
                    if descending:
                        axes = [a[::-1] for a in axes]
                    mesh = np.meshgrid(*axes, indexing='ij')
                    values = 2. + sum((i+1)*a for i, a in enumerate(mesh))
                    grid = tuple([np.array([7.])]*dropped + axes)
                    values = values.reshape((1,)*dropped + values.shape)

                    class Grid:
                        functions = ('synthetic_affine',)

                        def getGridValues(self, **kwargs):
                            return [(grid, values)]

                    n, removed, interpolator = createLayerFunctionInterpolators(False, 0, Grid())['synthetic_affine']
                    assert (n, removed) == (dimensions, dropped)
                    # Includes both boundaries and extrapolation on either side.
                    query = np.array([[-4.]*dimensions, [-2.]*dimensions,
                                      [.25]*dimensions, [3.]*dimensions, [5.]*dimensions])
                    expected = 2. + query @ np.arange(1, dimensions+1)
                    np.testing.assert_allclose(interpolator(query), expected, atol=2e-13)
                    interpolation_cases += 1
        with np.errstate(all='ignore'):
            singular = scorer.projectNeighborDisplacement(
                np.array([0.]), np.array([0.]), np.array([1.]),
                np.array([1.]), np.array([1.]), np.array([1.]))
        assert all(np.isnan(a).all() for a in singular)
    paths = [Path(__file__), ROOT/'sciona/trackml_bayesian.py', ROOT/'sciona/trackml_layer_functions.py']
    return dict(format='trackml-residual-numeric-semantics.v1', passed=True,
        approved=False, catalog_mutations=0, synthetic_only=True,
        projection_cases=projection_cases, error_cases=error_cases,
        interpolation_cases=interpolation_cases, singular_axis_nan_verified=True,
        native_threads=1, workers=1,
        runtime_versions={n: metadata.version(n) for n in ('numpy', 'scipy')},
        software_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        extraction_requirements=[
            'Projection must expose or document hemisphere orientation; conventional spherical polar projection alone changes the source below the equator.',
            'At z=0 the source polar projection is zero, not a unit tangent projection. The rotation axis is singular and must be bounded explicitly.',
            'Reusable uncertainty combination must receive explicit coefficients; empirical .007884, .2096 and .3853 remain recipe parameters.',
            'Generic regular-grid linear interpolation must preserve extrapolation and descending axes. Leading singleton removal and detector coordinate dispatch remain recipe policy.',
            'Inspect catalog candidates before adding new providers; this assessment does not establish catalog absence.',
            'Inject selected providers throughout ranking, extension and scoring, and repeat the full lifecycle and serialized graph checks before original CDG promotion.'],
        limitations=['Analytic synthetic boundary assessment only; no provider or CDG approval.',
                     'No real records or calibration maps; no historical accuracy claim.',
                     'Affine interpolation cases do not yet qualify arbitrary nonlinear grid values or malformed inputs.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_residual_numeric_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('software_sha256', 'extraction_requirements', 'limitations')}))

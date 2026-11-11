"""Synthetic source parity and independent covariance checks; no catalog writes."""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona import plasticc_observations as runtime


def validate(source_root: Path):
    pins_path = ROOT / 'docs/reviews/competition_plasticc_source_pins.json'
    pins = json.loads(pins_path.read_text())
    source = source_root / 'avocado/astronomical_object.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['files']['avocado/astronomical_object.py']
    instruments = source_root / 'avocado/instruments.py'
    assert hashlib.sha256(instruments.read_bytes()).hexdigest() == pins['files']['avocado/instruments.py']
    namespace = dict(vars(runtime))
    instrument_nodes = [n for n in ast.parse(instruments.read_text()).body if
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'band_central_wavelengths' for t in n.targets)
        or isinstance(n, ast.FunctionDef) and n.name == 'get_band_central_wavelength']
    exec(compile(ast.Module(body=instrument_nodes, type_ignores=[]), str(instruments), 'exec'), namespace)
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'AstronomicalObject')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    Reference = namespace['AstronomicalObject']
    counters = dict(preprocessing=0, fits=0, independent_covariance=0, likelihood_gradients=0,
                    optimizer_fallback=0, cached_fit=0, unknown_band=0)
    for seed in range(4):
        rng = np.random.default_rng(seed)
        times = np.sort(rng.uniform(0, 80, 30))
        table = pd.DataFrame(dict(time=times, band=np.resize(['lsstg', 'lssti', 'lsstu'], 30),
            flux=15 * np.sin(times / 17) + rng.normal(size=30), flux_error=rng.uniform(.4, 1.2, 30)))
        table.loc[5, 'flux'] += 70  # Synthetic robust-background outlier.
        table.index = np.arange(30) * 3 + 7  # idxmax is a label, not a position.
        original = table.copy(deep=True)
        obj = runtime.AstronomicalObject({'object_id': 'synthetic'}, table)
        ref = Reference({'object_id': 'synthetic'}, table)
        np.testing.assert_array_equal(obj.bands, ref.bands)
        np.testing.assert_array_equal(obj.bands, ['lsstu', 'lsstg', 'lssti'])
        pd.testing.assert_frame_equal(obj.subtract_background(), ref.subtract_background())
        pd.testing.assert_frame_equal(table, original)
        assert obj.preprocess_observations(subtract_background=False) is table
        counters['preprocessing'] += 1
        for fixed in (False, True):
            for background in (False, True):
                kwargs = dict(fix_scale=fixed, subtract_background=background)
                fitted, obs, params = obj.fit_gaussian_process(**kwargs)
                rfitted, robs, rparams = ref.fit_gaussian_process(**kwargs)
                pd.testing.assert_frame_equal(obs, robs)
                np.testing.assert_allclose(params, rparams, rtol=1e-10, atol=1e-10)
                query_times = np.linspace(-5, 90, 11)
                bands = ['lsstu', 'lsstr', 'lssti']
                mean, std = obj.predict_gaussian_process(bands, query_times, fitted_gp=fitted)
                rmean, rstd = ref.predict_gaussian_process(bands, query_times, fitted_gp=rfitted)
                np.testing.assert_allclose(mean, rmean, atol=1e-10)
                np.testing.assert_allclose(std, rstd, atol=1e-10)
                np.testing.assert_allclose(mean, obj.predict_gaussian_process(bands, query_times, uncertainties=False, fitted_gp=fitted))
                gp = fitted.func.__self__
                pars = gp.kernel.get_parameter_dict(include_frozen=True)
                # george's ConstantKernel sums the constant over its two axes.
                amplitude = 2 * np.exp(pars['k1:log_constant'])
                length2 = np.exp([pars['k2:metric:log_M_0_0'], pars['k2:metric:log_M_1_1']])
                def covariance(a, b):
                    r = np.sqrt(3 * np.sum((a[:, None] - b[None, :]) ** 2 / length2, axis=2))
                    return amplitude * (1 + r) * np.exp(-r)
                x = np.column_stack([obs.time, obs.band.map(runtime.get_band_central_wavelength)])
                q = np.array([[t, runtime.get_band_central_wavelength(b)] for b in bands for t in query_times])
                np.testing.assert_allclose(covariance(x, x), gp.kernel.get_value(x), rtol=1e-12, atol=1e-12)
                k = covariance(x, x) + np.diag(np.asarray(obs.flux_error) ** 2)
                cross = covariance(q, x)
                expected = cross @ np.linalg.solve(k, obs.flux)
                variance = amplitude - np.sum(cross * np.linalg.solve(k, cross.T).T, axis=1)
                np.testing.assert_allclose(mean.ravel(), expected, rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(std.ravel() ** 2, variance, rtol=1e-7, atol=1e-7)
                p = gp.get_parameter_vector().copy()
                numerical = []
                for i in range(len(p)):
                    delta = np.zeros_like(p); delta[i] = 1e-5
                    gp.set_parameter_vector(p + delta); plus = gp.log_likelihood(obs.flux)
                    gp.set_parameter_vector(p - delta); minus = gp.log_likelihood(obs.flux)
                    numerical.append((plus - minus) / 2e-5)
                gp.set_parameter_vector(p)
                np.testing.assert_allclose(gp.grad_log_likelihood(obs.flux), numerical, rtol=2e-4, atol=2e-5)
                counters['fits'] += 1; counters['independent_covariance'] += 1; counters['likelihood_gradients'] += 1
        def fail(fun, x, **kwargs):
            return SimpleNamespace(success=False, x=x + .25)
        with patch.object(runtime, 'minimize', fail):
            fallback, _, attempted = obj.fit_gaussian_process()
        with patch.dict(namespace, minimize=fail):
            reference_fallback, _, reference_attempted = ref.fit_gaussian_process()
        np.testing.assert_array_equal(attempted, reference_attempted)
        np.testing.assert_allclose(fallback.func.__self__.get_parameter_vector(), attempted - .25)
        np.testing.assert_allclose(fallback(q, return_cov=False), reference_fallback(q, return_cov=False))
        counters['optimizer_fallback'] += 1
        assert obj.get_default_gaussian_process() is obj.get_default_gaussian_process()
        counters['cached_fit'] += 1
    try:
        runtime.get_band_central_wavelength('synthetic-unknown')
    except runtime.AvocadoException:
        counters['unknown_band'] += 1
    else:
        raise AssertionError('Unknown band accepted')
    paths = [Path(runtime.__file__), Path(__file__), pins_path, ROOT / 'docs/licenses/Avocado-MIT.txt']
    return dict(status='passed', approved=False, scope='Synthetic numerical component only; full workflow remains incomplete',
        source_commit=pins['commit'], checks=counters,
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        dependencies={p: importlib.metadata.version(p) for p in ['astropy','george','numpy','scipy','pandas']},
        limitations=['No competition accuracy claim', 'No full graph execution yet',
                     'Source optimizer failure returns attempted parameters although predictor restores guesses'])


if __name__ == '__main__':
    report = validate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/private/tmp/sciona_plasticc_source'))
    (ROOT / 'docs/reviews/competition_plasticc_observations.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['checks'], sort_keys=True))

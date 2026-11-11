"""Validate complete source featurizer with synthetic GP and real GP integration."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
import warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona import plasticc_features as runtime
from sciona.plasticc_observations import AstronomicalObject


def reference(source_root):
    pins = json.loads((ROOT / 'docs/reviews/competition_plasticc_source_pins.json').read_text())
    namespace = {'np': SimpleNamespace(**{**vars(np), 'warnings': warnings}), 'pd': pd}
    for name in ['avocado/features.py', 'avocado/plasticc.py']:
        path = source_root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins['files'][name]
        nodes = [n for n in ast.parse(path.read_text()).body if
            isinstance(n, ast.ClassDef) and n.name in {'Featurizer', 'PlasticcFeaturizer'}
            or isinstance(n, ast.FunctionDef) and n.name == 'find_time_to_fractions'
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {'pad', 'plasticc_start_time', 'plasticc_end_time', 'plasticc_bands'} for t in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def same_dict(left, right):
    assert list(left) == list(right)
    for key in left:
        np.testing.assert_allclose(left[key], right[key], rtol=1e-10, atol=1e-10, equal_nan=True, err_msg=key)


def validate(source_root):
    ref = reference(source_root)
    featurizer = runtime.PlasticcFeaturizer()
    original = ref['PlasticcFeaturizer']()
    rng = np.random.default_rng(51)
    counters = dict(fraction_oracles=0, synthetic_model_cases=0, independent_statistics=0,
                    selected_feature_cases=0, dataframe_selection=0, real_gp_integration=0,
                    leakage_exclusion=0)
    # Independent threshold oracle: strict inequality, first tied maximum, no crossing -> NaN.
    for flux in [np.array([1., 5., 4., 2.5, 1., 0.]), np.array([0., 4., 4., 0.]),
                 np.ones(7), -np.arange(1., 8.), np.zeros(7), *rng.normal(size=(10, 20))]:
        for forward in (True, False):
            fractions = [.8, .5, .2]
            peak = int(np.argmax(flux))
            indices = range(peak + 1, len(flux)) if forward else range(peak - 1, -1, -1)
            indices = list(indices)
            expected = [next((abs(i - peak) for i in indices if flux[i] < flux[peak] * f), np.nan) for f in fractions]
            actual = runtime.find_time_to_fractions(flux, fractions, forward)
            np.testing.assert_allclose(actual, expected, equal_nan=True)
            np.testing.assert_allclose(actual, ref['find_time_to_fractions'](flux, fractions, forward), equal_nan=True)
            counters['fraction_oracles'] += 1
    rows = []
    for case in range(6):
        start = runtime.plasticc_start_time
        t = np.arange(start - runtime.pad, runtime.plasticc_end_time + runtime.pad + 1)
        phase = t - start
        model = np.array([(10 + b) * np.sin(phase / (25 + case * 12) + b / 8) +
                         25 * np.exp(-((phase - 400 - b * 2) / 45) ** 2) for b in range(6)])
        if case == 1: model = -np.abs(model)
        if case == 2: model = np.abs(model)
        if case == 3: model[:] = 1  # Source undefined ratios/peak fractions retained.
        bands = runtime.plasticc_bands[:3] if case == 4 else runtime.plasticc_bands
        obs = pd.DataFrame(dict(time=np.sort(rng.uniform(start, runtime.plasticc_end_time, 72)),
             band=np.resize(bands, 72), flux=rng.normal(0, 12, 72), flux_error=rng.uniform(.5, 2, 72)))
        metadata = dict(host_specz=.2, host_photoz=.23, host_photoz_error=.04,
                        ra=0., decl=0., mwebv=.01, ddf=bool(case % 2))
        class SyntheticObject:
            def __init__(self): self.metadata = metadata
            def fit_gaussian_process(self): return None, obs, np.array([2., 3.])
            def predict_gaussian_process(self, requested_bands, times, **kwargs):
                assert requested_bands == runtime.plasticc_bands
                np.testing.assert_array_equal(times, t)
                return model.copy()
        obj = SyntheticObject()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            raw, returned_model = featurizer.extract_raw_features(obj, return_model=True)
            expected, expected_model = original.extract_raw_features(obj, return_model=True)
            same_dict(raw, expected)
            pd.testing.assert_frame_equal(returned_model, expected_model)
            same_dict(raw, featurizer.extract_raw_features(obj))
            selected = featurizer.select_features(raw)
            same_dict(selected, original.select_features(expected))
            same_dict(selected, featurizer.extract_features(obj))
        # Independently derive observation statistics and model extrema.
        assert raw['count'] == len(obs)
        s2n = obs.flux.to_numpy() / obs.flux_error.to_numpy()
        np.testing.assert_allclose(raw['frac_background'], np.mean(np.abs(s2n) < 3))
        for threshold in [-20, -10, -5, -3, 3, 5, 10, 20]:
            assert raw['count_s2n_%d' % threshold] == sum(v < threshold if threshold < 0 else v > threshold for v in s2n)
        for b, band in enumerate(runtime.plasticc_bands):
            assert raw['max_flux_' + band] == max(model[b])
            assert raw['min_flux_' + band] == min(model[b])
            np.testing.assert_allclose(raw['abs_diff_' + band], sum(abs(float(y) - float(x)) for x, y in zip(model[b, :-1], model[b, 1:])))
            mask = obs.band == band
            np.testing.assert_allclose(raw['total_s2n_' + band], np.linalg.norm(s2n[mask]))
        assert not {'host_specz', 'ra', 'decl', 'ddf', 'mwebv'}.intersection(selected)
        changed = dict(raw, host_specz=99., ra=99., decl=-99., ddf=not metadata['ddf'], mwebv=99.)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            same_dict(selected, featurizer.select_features(changed))
        rows.append(raw)
        counters['synthetic_model_cases'] += 1; counters['independent_statistics'] += 1
        counters['selected_feature_cases'] += 1; counters['leakage_exclusion'] += 1
    frame = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        selected_frame = featurizer.select_features(frame)
        pd.testing.assert_frame_equal(selected_frame, original.select_features(frame))
        for i, row in enumerate(rows):
            same_dict(selected_frame.iloc[i].to_dict(), featurizer.select_features(row))
    counters['dataframe_selection'] += 1
    # Actual astropy/george pipeline, full source grid and every band.
    for seed in range(2):
        times = np.sort(rng.uniform(start, start + 130, 48))
        obs = pd.DataFrame(dict(time=times, band=np.resize(runtime.plasticc_bands, 48),
             flux=20 * np.sin((times - start) / 20) + rng.normal(size=48), flux_error=np.ones(48)))
        obj = AstronomicalObject(dict(metadata, object_id='synthetic'), obs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            actual = featurizer.extract_features(obj)
            expected = original.extract_features(obj)
            same_dict(actual, expected)
        counters['real_gp_integration'] += 1
    paths = [Path(runtime.__file__), Path(__file__), ROOT / 'sciona/plasticc_observations.py',
             ROOT / 'docs/reviews/competition_plasticc_source_pins.json', ROOT / 'docs/licenses/Avocado-MIT.txt']
    return dict(status='passed', approved=False, checks=counters, raw_feature_count=len(rows[0]),
        selected_feature_count=len(selected_frame.columns),
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Complete numerical featurizer; synthetic checks only; full workflow still incomplete',
        adaptations=['numpy.warnings replaced by standard warnings; reference uses isolated alias'],
        limitations=['Source NaN and nonfinite ratio semantics preserved',
                     'No competition accuracy or full CDG claim'])


if __name__ == '__main__':
    result = validate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/private/tmp/sciona_plasticc_source'))
    (ROOT / 'docs/reviews/competition_plasticc_features.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'hashes'}, indent=2))

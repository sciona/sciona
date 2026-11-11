"""Synthetic, pinned-source augmentation checks; no empirical catalogs loaded."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona import plasticc_augmentation as runtime


def source_class(root, empirical, seed, retries):
    pins = json.loads((ROOT / 'docs/reviews/competition_plasticc_source_pins.json').read_text())
    namespace = dict(vars(runtime))
    namespace['np'] = SimpleNamespace(**{**vars(np), 'random': np.random.RandomState(seed)})
    namespace['settings'] = {'augment_retries': retries}
    for name, cls in [('avocado/augment.py', 'Augmentor'), ('avocado/plasticc.py', 'PlasticcAugmentor')]:
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins['files'][name]
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == cls)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    cls = namespace['PlasticcAugmentor']
    # Only substitute the external file boundary. All numerical methods remain original.
    cls._load_photoz_reference = lambda self: empirical.copy()
    return cls(), namespace['np'].random


def validate(root):
    empirical = np.array([[.1, .12, .02], [.3, .2, .04], [.8, 1.2, .1]])
    counters = dict(metadata=0, sampling=0, noise=0, detection=0, object_augmentation=0,
                    photoz=0, retry_exhaustion=0, validation_rejections=0, real_gp=0)
    for seed in range(8):
        for galactic, ddf in [(True, False), (True, True), (False, False), (False, True)]:
            actual = runtime.PlasticcAugmentor(empirical, seed=seed, augment_retries=3)
            ref, ref_rng = source_class(root, empirical, seed, 3)
            z = 0 if galactic else .3
            metadata = dict(object_id='synthetic', redshift=z, host_specz=z, host_photoz=z,
                            host_photoz_error=.03, galactic=galactic, ddf=ddf, mwebv=.01)
            times = np.linspace(0, 1100, 480)
            obs = pd.DataFrame(dict(time=times, band=np.resize(['lsstu','lsstg','lsstr','lssti','lsstz','lssty'], len(times)),
                 flux=1000 + 100 * np.sin(times / 30), flux_error=np.ones(len(times))))
            obj = runtime.AstronomicalObject(metadata, obs)
            # Analytic synthetic predictor isolates cosmology, wavelength and resampling logic.
            def gp(x, return_var=True):
                return 1000 + 100 * np.sin(x[:, 0] / 30) + x[:, 1] / 100, np.full(len(x), 4.)
            obj._default_gaussian_process = gp
            a = actual._augment_metadata(obj); b = ref._augment_metadata(obj)
            assert a == b; counters['metadata'] += 1
            if galactic:
                assert a['host_specz'] == a['host_photoz'] == a['redshift'] == 0
            else:
                assert .95 * z <= a['redshift'] <= min(5*z, 1.5*(1+z)-1)
            aobs = actual._choose_sampling_times(obj, a)
            bobs = ref._choose_sampling_times(obj, b)
            pd.testing.assert_frame_equal(aobs, bobs); counters['sampling'] += 1
            # Whole resampling exercises brightness scaling, errors and detection.
            ar = actual._resample_light_curve(obj, a); br = ref._resample_light_curve(obj, b)
            assert ar is not None and br is not None
            pd.testing.assert_frame_equal(ar, br)
            scale = 10 ** (-.4 * a['augment_brightness'])
            if z:
                scale *= 10 ** (.4 * (actual.cosmology.distmod(z) - actual.cosmology.distmod(a['redshift'])).value)
            wavelength = ar.band.map(runtime.get_band_central_wavelength).to_numpy() / ((1+a['redshift'])/(1+z))
            expected = (1000 + 100 * np.sin(ar.reference_time.to_numpy()/30) + wavelength/100)*scale
            np.testing.assert_allclose(ar.model_flux, expected)
            np.testing.assert_allclose(ar.model_flux_error, 2*scale)
            assert (ar.flux_error >= ar.model_flux_error).all()
            counters['noise'] += 1; counters['detection'] += 1
            aa = actual.augment_object(obj, force_success=False)
            bb = ref.augment_object(obj, force_success=False)
            assert aa is not None and bb is not None
            assert aa.metadata == bb.metadata
            pd.testing.assert_frame_equal(aa.observations, bb.observations)
            counters['object_augmentation'] += 1
            # Replays include rejection sampling of negative photo-z proposals.
            for redshift in [.001, .1, .5]:
                np.testing.assert_array_equal(actual._simulate_photoz(redshift), ref._simulate_photoz(redshift))
                counters['photoz'] += 1
            np.testing.assert_array_equal(actual.rng.get_state()[1], ref_rng.get_state()[1])
            assert actual.rng.get_state()[2:] == ref_rng.get_state()[2:]
            pd.testing.assert_frame_equal(obj.observations, obs)
    # Deterministic failure path: every detection fails, exactly the requested retries.
    actual = runtime.PlasticcAugmentor(empirical, seed=2, augment_retries=3)
    with patch.object(actual, '_simulate_detection', side_effect=lambda o,m: (o,False)) as detector:
        assert actual.augment_object(obj, force_success=False) is None
        assert detector.call_count == 3
    counters['retry_exhaustion'] += 1
    for bad in [[], [[.1,.2]], [[0,.1,.2]], [[.1,-.2,.3]], [[.1,.2,np.nan]]]:
        try: runtime.PlasticcAugmentor(bad, seed=1, augment_retries=3)
        except ValueError: counters['validation_rejections'] += 1
        else: raise AssertionError('Bad reference accepted')
    for bad in [0,-1,True,1.5]:
        try: runtime.PlasticcAugmentor(empirical, seed=1, augment_retries=bad)
        except ValueError: counters['validation_rejections'] += 1
        else: raise AssertionError('Bad retry count accepted')
    # Real GP integration with high synthetic signal; no analytic predictor replacement.
    real_obj = runtime.AstronomicalObject(dict(metadata, galactic=True, host_specz=0,redshift=0),
        obs.iloc[::4].reset_index(drop=True))
    actual = runtime.PlasticcAugmentor(empirical, seed=9, augment_retries=3)
    ref, _ = source_class(root, empirical, 9, 3)
    aa = actual.augment_object(real_obj, force_success=False)
    bb = ref.augment_object(real_obj, force_success=False)
    assert aa is not None and bb is not None
    assert aa.metadata == bb.metadata
    pd.testing.assert_frame_equal(aa.observations, bb.observations)
    counters['real_gp'] += 1
    paths = [Path(runtime.__file__), Path(__file__), ROOT/'sciona/plasticc_observations.py',
             ROOT/'docs/reviews/competition_plasticc_source_pins.json', ROOT/'docs/licenses/Avocado-MIT.txt']
    return dict(status='passed', approved=False, checks=counters,
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Object augmentation only, including actual GP integration; synthetic reference only',
        adaptations=['Explicit empirical reference replaces implicit file read',
                     'Instance RandomState replaces global RandomState; sequence parity checked',
                     'Retry count explicit; no guessed settings default'],
        remaining=['Dataset orchestration and source settings', 'Training and full graph execution'])

if __name__ == '__main__':
    report = validate(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/private/tmp/sciona_plasticc_source'))
    (ROOT/'docs/reviews/competition_plasticc_augmentation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],sort_keys=True))

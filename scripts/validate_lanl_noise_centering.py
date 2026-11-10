"""Qualify the explicit noise/median operation on synthetic full-size windows.

Reference is the equation in the winner's prose, not recovered executable
source. RNG and noise alignment are explicit test policy, not historical claims.
"""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from sciona.atoms.ml import noise_median_centering as provider

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    provider_path = Path(inspect.getfile(provider))
    test_path = ROOT.parent/'sciona-atoms-ml/tests/test_noise_median_centering_provider.py'
    before = {str(p.relative_to(ROOT.parent)): sha(p) for p in (provider_path, test_path, Path(__file__))}
    rng = np.random.RandomState(457)
    width = 150_000
    zero = np.zeros((2, width), dtype=np.int16)
    quantized = rng.randint(-20, 21, (2, width)).astype(np.int16)
    oscillator = np.tile(4*np.sin(np.arange(width)/31), (2, 1))
    impulses = zero.copy()
    impulses[:, ::997] = 30000
    cases = []
    for name, windows in [('constant', zero), ('quantized', quantized),
                          ('oscillatory', oscillator), ('sparse_impulses', impulses)]:
        noise = rng.normal(0, .5, windows.shape)
        expected = []
        for signal, perturbation in zip(windows, noise):
            value = signal + perturbation
            expected.append(value - np.median(value))
        actual = provider.add_noise_median_center(windows, noise, 'synthetic-amplitude.v1')
        np.testing.assert_array_equal(actual, np.array(expected))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(np.median(actual, axis=1), 0., atol=1e-14, rtol=0)
        cases.append(dict(synthetic_family=name, windows=2, samples_per_window=width,
                          exact_prose_equation_equal=True, finite=True))
    if before != {str(p.relative_to(ROOT.parent)): sha(p) for p in (provider_path, test_path, Path(__file__))}:
        raise ValueError('Implementation changed during validation')
    return dict(format='lanl-noise-centering-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, cases=cases,
        source='https://www.kaggle.com/c/LANL-earthquake-Prediction/discussion/94390',
        reference_kind='Independent implementation of the explicitly described arithmetic, not source-code replay.',
        software_sha256=before, resource_policy=dict(concurrent_workers=1, native_threads=1),
        reuse='Caller supplies aligned noise in the same amplitude coordinate; window geometry, distribution, seed and event semantics stay outside the operation.',
        limitations=['This component removes absolute baseline and adds perturbations; predictive improvement is not established.',
            'The selected synthetic RNG seed and per-window noise draws do not recover historical random-state or noise reuse policy.',
            'Feature extraction, distribution screening, all model families and final blend remain unqualified.',
            'Provider registration alone does not approve catalog selection or the original competition CDG.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/lanl_noise_centering_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, cases=len(report['cases']), approved=False)))

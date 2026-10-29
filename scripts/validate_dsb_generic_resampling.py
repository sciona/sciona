"""Compare reusable resampling to the hash-verified public preparation source."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom
from sciona.atoms.numerical.spatial import resampling as candidate
from scripts.audit_competition_dsb_semantics import checked_source

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Explicit single-thread environment required')
    pins = json.loads((ROOT / 'docs/reviews/competition_dsb_source_pins.json').read_text())
    name = 'training/prepare.py'
    tree = ast.parse(checked_source(source_root, pins, name))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resample']
    if len(functions) != 1:
        raise ValueError('Source resampling inventory changed')
    scope = dict(np=np, zoom=zoom)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-public-resampling>', 'exec'), scope)
    cases = 0
    for dtype in (np.uint8, np.int16, np.float32, np.float64):
        for shape in ((3, 4, 5), (1, 3, 2)):
            volume = (np.arange(np.prod(shape)).reshape(shape) % 23).astype(dtype)
            for spacing in ([1., 1., 1.], [1.5, 2., .8]):
                for order in range(6):
                    spacing = np.asarray(spacing)
                    requested = np.ones(3)
                    expected, expected_spacing = scope['resample'](volume, spacing, requested, order)
                    actual, actual_spacing = candidate.resample_regular_volume(volume, spacing, requested, order)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual_spacing, expected_spacing)
                    assert actual.dtype == expected.dtype
                    cases += 1
    return dict(passed=True, approved=False, synthetic_only=True, catalog_mutations=0,
                source_commit=pins['commit'], source_hashes={name:pins['files'][name]},
                cases=cases, exact_arrays=True, exact_effective_spacing=True, native_threads=1,
                provider_sha256=sha(Path(inspect.getfile(candidate))), validator_sha256=sha(Path(__file__)),
                source_extractor_sha256=sha(ROOT / 'scripts/audit_competition_dsb_semantics.py'),
                test_sha256=sha(ROOT / 'tests/test_regular_volume_resampling.py'),
                limitations=['Three-dimensional arrays only; channel iteration is caller policy.',
                             'Current SciPy interpolation semantics qualified, not historical library behavior.',
                             'Preparation integration and catalog approval remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    result = validate(parser.parse_args().source_root)
    (ROOT / 'docs/reviews/dsb_generic_resampling_source_parity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

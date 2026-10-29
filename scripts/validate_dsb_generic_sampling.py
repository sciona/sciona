"""Compare reusable sampling atoms directly with hash-pinned public source.

Only synthetic arrays are used. No graph or provider is catalog-approved here.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from sciona.atoms.numerical.statistics import threshold_sampling as candidate
from scripts.audit_competition_dsb_semantics import checked_source, source_oversampling

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_state(a, b):
    if a[0] != b[0] or a[2:] != b[2:] or not np.array_equal(a[1], b[1]):
        raise ValueError('RNG continuation differs from pinned source')


def validate(source_root):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Explicit single-thread environment required')
    pins = json.loads((ROOT/'docs/reviews/competition_dsb_source_pins.json').read_text())
    sampling = checked_source(source_root, pins, 'data_classifier.py')
    tree = ast.parse(sampling[sampling.index('def sample('):])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if [node.name for node in functions] != ['sample', 'sampleone', 'softmax', 'augment']:
        raise ValueError('Unexpected source sampler inventory')
    tree.body = [node for node in functions if node.name in {'sample', 'sampleone', 'softmax'}]
    scope = dict(np=np, range=lambda *args: list(range(*args)))
    exec(compile(tree, '<pinned-public-sampler-python3>', 'exec'), scope)
    global_state = np.random.get_state()
    sampling_cases = 0
    try:
        for scores in [np.array([0., -1000., -2000.]), np.array([1.2, -3., 5., .6]), np.zeros(5)]:
            for temperature in [.3, .7, 2.]:
                for k in [0, 1, len(scores)-1, len(scores), len(scores)+1]:
                    for seed in [0, 7, 91]:
                        np.random.seed(seed)
                        expected = scope['sample'](scores.copy(), k, temperature)
                        expected_state = np.random.get_state()
                        rng = np.random.RandomState(seed)
                        actual = candidate.sequential_softmax_indices(scores, k, temperature, rng=rng)
                        np.testing.assert_array_equal(actual, expected)
                        same_state(rng.get_state(), expected_state)
                        same_state(np.random.get_state(), expected_state)
                        sampling_cases += 1
    finally:
        np.random.set_state(global_state)
    detector = checked_source(source_root, pins, 'data_detector.py')
    repetition_cases = 0
    for sizes in [[5., 6., 7., 30., 31., 40., 41.], [50., 10., 35., 6.], [7.], [6., 7.]]:
        rows = np.column_stack((np.arange(len(sizes)), np.zeros((len(sizes), 2)), sizes))
        expected = source_oversampling(detector, [rows])[:, 1:]
        actual = candidate.repeat_rows_by_threshold(rows, 3, [6., 30., 40.], [1, 2, 4])
        np.testing.assert_array_equal(actual, expected)
        repetition_cases += 1
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_commit=pins['commit'], source_hashes={n:pins['files'][n] for n in ['data_classifier.py','data_detector.py']},
        sampling_cases=sampling_cases, repetition_cases=repetition_cases,
        exact_selected_indices=True, exact_rng_continuation=True, exact_repeated_rows=True,
        global_rng_restored=True, native_threads=1,
        provider_sha256=sha(Path(inspect.getfile(candidate))), validator_sha256=sha(Path(__file__)),
        source_extractor_sha256=sha(ROOT/'scripts/audit_competition_dsb_semantics.py'),
        synthetic_test_sha256=sha(ROOT/'tests/test_threshold_sampling.py'),
        adaptations=['Python 2 mutable range materialized as a list; sampler arithmetic unchanged.'],
        limitations=['Source comparison uses the source probability floor 1e-5 and thresholds 6,30,40 with additive multiplicities 1,2,4.',
            'Other supported numerical inputs are covered by separate synthetic contract tests, not this source comparison.',
            'No full graph, model training, predictive-quality or catalog publication claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source_root)
    (ROOT/'docs/reviews/dsb_generic_sampling_source_parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

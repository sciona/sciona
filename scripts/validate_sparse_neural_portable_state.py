"""Train all declared sparse models and replay numeric state in a fresh process."""
# Keep TensorFlow initialization ahead of sklearn/pandas/Arrow.
from sciona.sparse_neural_regression import SparseRegressor, QuantileSparseRegressor, prelu
from sciona.sparse_neural_state import export_prediction_state, restore_prediction_state

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.sparse import csr_matrix
from inventory_mercari_models import inventory

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(runtime, expected_manifest):
    if sha(runtime/'manifest.json') != expected_manifest:
        raise ValueError('Synthetic replay manifest changed')
    manifest = json.loads((runtime/'manifest.json').read_text())
    required = {'query.npz'} | {f'state-{i}.json' for i in range(12)}
    if set(manifest['files']) != required:
        raise ValueError('Exact twelve-model bundle required')
    for relative, digest in manifest['files'].items():
        path = runtime/relative
        if path.is_symlink() or sha(path) != digest:
            raise ValueError('Synthetic replay content changed')
    with np.load(runtime/'query.npz', allow_pickle=False) as arrays:
        Q, expected = csr_matrix(arrays['query']), arrays['expected']
    for i in range(12):
        state = json.loads((runtime/f'state-{i}.json').read_text())
        model = restore_prediction_state(state, [f'feature-{j}' for j in range(Q.shape[1])],
                                         'synthetic-feature.v1', 'synthetic-response.v1')
        try:
            if hasattr(model, '_train_op'):
                raise ValueError('Prediction restoration unexpectedly constructed a training operation')
            np.testing.assert_array_equal(model.predict(Q), expected[i])
        finally:
            model.close()
    return dict(passed=True, restored_models=12, exact_predictions=True, training_executed=False,
                executable_deserialization=False)


def validate(directory):
    paths = ['sciona/sparse_neural_regression.py', 'sciona/sparse_neural_state.py',
             'sciona/quantile_target_encoding.py', 'scripts/validate_sparse_neural_portable_state.py',
             'scripts/inventory_mercari_models.py', 'tests/test_sparse_neural_state.py']
    before = {p: sha(ROOT/p) for p in paths}
    reference_path = 'docs/reviews/sparse_neural_reference.json'
    pinned = subprocess.check_output(['git', 'show', 'b94f91c9:'+reference_path], cwd=ROOT)
    if pinned != (ROOT/reference_path).read_bytes():
        raise ValueError('Immutable source-parity review changed')
    reference = json.loads(pinned)
    if not reference['passed'] or reference['software_sha256']['sciona/sparse_neural_regression.py'] != before['sciona/sparse_neural_regression.py']:
        raise ValueError('Source-qualified model core required')
    recipes = inventory(directory)
    rng = np.random.default_rng(881)
    dense = rng.uniform(0, 1, (8192, 23)).astype(np.float32)
    dense[rng.random(dense.shape) < .7] = 0
    X = csr_matrix(dense)
    y = np.log1p(1+dense[:, 0]+2*dense[:, 1]+rng.uniform(0, .2, len(dense)))
    query = rng.uniform(0, 1, (37, 23)).astype(np.float32)
    query[rng.random(query.shape) < .7] = 0
    Q = csr_matrix(query)
    expected, cases = [], []
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-neural-state-') as temporary:
        runtime = Path(temporary)
        for i, record in enumerate(recipes['variants']['tensorflow']):
            controls = dict(record['resolved_constructor_controls'], actfunc=prelu)
            cls = QuantileSparseRegressor if record['model_class'] == 'RegressionClf' else SparseRegressor
            model = cls(**controls)
            try:
                faulthandler.dump_traceback_later(120, exit=True)
                model.fit(X, y)
                expected.append(model.predict(Q))
                state = export_prediction_state(model, [f'feature-{j}' for j in range(X.shape[1])],
                                                'synthetic-feature.v1', 'synthetic-response.v1')
                (runtime/f'state-{i}.json').write_text(json.dumps(state, allow_nan=False))
                cases.append(dict(group=record['group'], seed=record['seed'],
                                  mode=state['mode'], optimizer_updates=model.update_count_))
            finally:
                faulthandler.cancel_dump_traceback_later()
                model.close()
        if len(cases) != 12 or sum(c['optimizer_updates'] for c in cases) != 72:
            raise ValueError('Complete declared training schedule required')
        np.savez(runtime/'query.npz', query=query, expected=np.stack(expected))
        manifest = dict(files={p.name: sha(p) for p in sorted(runtime.iterdir())})
        (runtime/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--replay', str(runtime),
            '--manifest-sha256', sha(runtime/'manifest.json')], capture_output=True, text=True, timeout=120,
            env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
                 'VECLIB_MAXIMUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1',
                 'TF_NUM_INTRAOP_THREADS':'1', 'TF_NUM_INTEROP_THREADS':'1'})
        if child.returncode:
            raise ValueError('Fresh-process numeric replay failed: '+child.stderr[-2000:])
        child_result = json.loads(child.stdout.strip().splitlines()[-1])
        if child_result != dict(passed=True, restored_models=12, exact_predictions=True,
                                training_executed=False, executable_deserialization=False):
            raise ValueError('Complete fresh-process replay required')
    if before != {p: sha(ROOT/p) for p in paths}:
        raise ValueError('Implementation changed during qualification')
    return dict(format='sparse-neural-portable-state.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, model_fits=12, optimizer_updates=72, cases=cases,
        fresh_process_replay=child_result, software_sha256=before,
        source_parity_commit='b94f91c9', source_parity_sha256=hashlib.sha256(pinned).hexdigest(),
        resource_policy=dict(concurrent_fits=1, intra_op_threads=1, inter_op_threads=1,
            training_worker_processes=0, replay_child_processes=1, parent_waits_during_child_replay=True),
        limitations=['Prediction-only JSON state excludes optimizer state; training resumption is not supported.',
            'Exact replay covers the declared twelve synthetic configurations in the current runtime; clean deployment and other platforms remain unqualified.',
            'Feature and response coordinate labels are caller declarations, not proof of matching semantics.',
            'Application feature graphs are bypassed; no original CDG approval or real-data accuracy claim.',
            'Numeric learned state is private runtime material and is not included in this report.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.replay, args.manifest_sha256)))
    else:
        if not args.source_directory:
            parser.error('--source-directory is required for training validation')
        result = validate(args.source_directory)
        (ROOT/'docs/reviews/sparse_neural_portable_state.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(passed=True, model_fits=12, fresh_process_replay=result['fresh_process_replay'])))

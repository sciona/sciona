"""Compare complete declared model schedules with reusable sparse neural code."""
# Initialize TensorFlow before other numerical dependencies.
from probe_mercari_tensorflow_runtime import load_model_namespace
from sciona.sparse_neural_regression import SparseRegressor, QuantileSparseRegressor, prelu

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory):
    paths = ['sciona/sparse_neural_regression.py', 'sciona/quantile_target_encoding.py',
             'scripts/validate_sparse_neural_reference.py', 'scripts/probe_mercari_tensorflow_runtime.py',
             'scripts/inventory_mercari_models.py', 'docs/licenses/mercari-solution-LICENSE.txt']
    before = {p: sha(ROOT/p) for p in paths}
    namespace, inventory, quantile_sha = load_model_namespace(directory)
    rng = np.random.default_rng(877)
    dense = rng.uniform(0, 1, (8192, 23)).astype(np.float32)
    dense[rng.random(dense.shape) < .7] = 0
    X = csr_matrix(dense)
    y = np.log1p(1 + dense[:, 0] + 2*dense[:, 1] + rng.uniform(0, .2, len(dense)))
    q = rng.uniform(0, 1, (37, 23)).astype(np.float32)
    q[rng.random(q.shape) < .7] = 0
    Q = csr_matrix(q)
    cases = []
    for record in inventory['variants']['tensorflow']:
        controls = dict(record['resolved_constructor_controls'])
        source = namespace[record['model_class']](**dict(controls, actfunc=namespace['prelu']))
        cls = QuantileSparseRegressor if record['model_class'] == 'RegressionClf' else SparseRegressor
        candidate = cls(**dict(controls, actfunc=prelu))
        try:
            faulthandler.dump_traceback_later(120, exit=True)
            source.fit(X, y, use_gpu=False, verbose=False)
            candidate.fit(X, y)
            expected, actual = source.predict(Q), candidate.predict(Q)
            np.testing.assert_array_equal(actual, expected)
            expected_updates = sum((len(y)+controls['batch_size']*2**min(epoch, 2)-1)
                                   // (controls['batch_size']*2**min(epoch, 2)) for epoch in range(3))
            assert source.update_count_ == candidate.update_count_ == expected_updates
            cases.append(dict(group=record['group'], seed=record['seed'],
                source_class=record['model_class'], reusable_class=cls.__name__,
                exact_predictions=True, optimizer_updates_per_implementation=expected_updates,
                epoch_schedule=candidate.epoch_schedule_))
        finally:
            faulthandler.cancel_dump_traceback_later()
            candidate.close()
            if hasattr(source, '_session'):
                source._session.close()
        print(json.dumps(dict(completed_model_pairs=len(cases))), flush=True)
    if len(cases) != 12 or sum(c['optimizer_updates_per_implementation'] for c in cases) != 72:
        raise ValueError('Complete declared model schedule required')
    if before != {p: sha(ROOT/p) for p in paths}:
        raise ValueError('Implementation changed during comparison')
    return dict(format='sparse-neural-reference.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, model_pairs=12, fits=24, optimizer_updates=144, cases=cases,
        software_sha256=before, source_commit=inventory['source_commit'],
        source_sha256={**inventory['source_sha256'], 'mercari/regression_clf.py': quantile_sha},
        resource_policy=dict(concurrent_fits=1, session_cpu_devices=1, intra_op_threads=1, inter_op_threads=1),
        limitations=['Current-runtime synthetic sparse-feature comparison; the application feature pipelines are bypassed.',
            'All twelve declared full-width, three-epoch schedules are compared; historical winner backend and additional models remain unresolved.',
            'Source Huber-named models use the MSE implementation, preserving actual method dispatch.',
            'Response transform, feature identity and quantile precision units remain caller obligations.',
            'No catalog publication or portable model-state qualification. No real-data accuracy or historical-runtime numerical claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    result = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/sparse_neural_reference.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, model_pairs=result['model_pairs'], optimizer_updates=result['optimizer_updates'])))

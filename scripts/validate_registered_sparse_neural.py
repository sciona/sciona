"""Compare registered isolated sparse-model operations with the qualified core."""
from sciona.sparse_neural_regression import SparseRegressor, QuantileSparseRegressor, prelu
from sciona.sparse_neural_state import export_prediction_state
from sciona.atoms.ml import sparse_neural_prediction as provider

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.sparse import csr_matrix
from inventory_mercari_models import inventory

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory):
    paths = ['sciona/sparse_neural_regression.py', 'sciona/sparse_neural_state.py',
             'sciona/sparse_neural_worker.py', 'sciona/sparse_neural_process.py',
             'sciona/quantile_target_encoding.py', 'scripts/validate_registered_sparse_neural.py',
             'scripts/inventory_mercari_models.py']
    before = {p: sha(ROOT/p) for p in paths}
    provider_sha = sha(provider.__file__)
    evidence_path = 'docs/reviews/sparse_neural_portable_state.json'
    committed = subprocess.check_output(['git', 'show', '759bed67:'+evidence_path], cwd=ROOT)
    if committed != (ROOT/evidence_path).read_bytes():
        raise ValueError('Immutable portable-state qualification changed')
    evidence = json.loads(committed)
    for p in ('sciona/sparse_neural_regression.py', 'sciona/sparse_neural_state.py', 'sciona/quantile_target_encoding.py'):
        if before[p] != evidence['software_sha256'][p]:
            raise ValueError('Qualified model implementation changed')
    recipes = inventory(directory)
    rng = np.random.default_rng(883)
    dense = rng.uniform(0, 1, (8192, 23)).astype(np.float32)
    dense[rng.random(dense.shape) < .7] = 0
    X = csr_matrix(dense)
    y = np.log1p(1+dense[:, 0]+2*dense[:, 1]+rng.uniform(0, .2, len(dense)))
    query = rng.uniform(0, 1, (37, 23)).astype(np.float32)
    query[rng.random(query.shape) < .7] = 0
    Q = csr_matrix(query)
    columns = [f'feature-{j}' for j in range(X.shape[1])]
    cases = []
    for record in recipes['variants']['tensorflow']:
        mode = 'quantile' if record['model_class'] == 'RegressionClf' else 'mse'
        controls = dict(record['resolved_constructor_controls'])
        if controls.pop('actfunc') != {'source_expression': 'prelu'}:
            raise ValueError('Expected source PReLU activation')
        controls['n_hidden'] = list(controls['n_hidden'])
        controls.update(activation='prelu', quantile_precision=160., classification_loss_multiplier=6.)
        baseline_controls = dict(controls)
        baseline_controls.pop('activation')
        cls = QuantileSparseRegressor if mode == 'quantile' else SparseRegressor
        model = cls(**baseline_controls, actfunc=prelu)
        try:
            faulthandler.dump_traceback_later(120, exit=True)
            model.fit(X, y)
            expected = model.predict(Q)
            expected_state = export_prediction_state(model, columns, 'synthetic-features.v1', 'synthetic-response.v1')
            updates = model.update_count_
        finally:
            faulthandler.cancel_dump_traceback_later()
            model.close()
        state = provider.fit_sparse_neural(X, y, columns, 'synthetic-features.v1', 'synthetic-response.v1', mode, controls)
        state = json.loads(json.dumps(state, allow_nan=False))
        if state != expected_state:
            raise ValueError('Registered fit differs from the qualified direct model state')
        actual = provider.predict_sparse_neural(Q, columns, 'synthetic-features.v1', 'synthetic-response.v1', state)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        cases.append(dict(group=record['group'], seed=record['seed'], mode=mode,
            optimizer_updates_per_implementation=updates, exact_numeric_state=True, exact_predictions=True,
            worker_full_schedule_verified=True))
        print(json.dumps(dict(completed_pairs=len(cases))), flush=True)
    if len(cases) != 12 or sum(c['optimizer_updates_per_implementation'] for c in cases) != 72:
        raise ValueError('Full twelve-model training coverage required')
    if before != {p: sha(ROOT/p) for p in paths} or provider_sha != sha(provider.__file__):
        raise ValueError('Implementation changed during validation')
    return dict(format='registered-sparse-neural-reference.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, model_pairs=12, fits=24, optimizer_updates=144, cases=cases,
        software_sha256=before, provider_sha256=provider_sha,
        portable_evidence_commit='759bed67', portable_evidence_sha256=hashlib.sha256(committed).hexdigest(),
        resource_policy=dict(concurrent_fits=1, child_workers=1, native_pool_threads=1,
                             parent_waits_during_child_compute=True),
        limitations=['Application feature graphs are bypassed; full connected registered-provider execution remains required.',
            'Known model architecture and numeric JSON weights only; no optimizer-state resumption.',
            'Exact current-runtime synthetic parity does not establish historical winner completeness or new-domain accuracy.',
            'Catalog intake, dependency qualification and Tier 3 publication remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    result = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/registered_sparse_neural_reference.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, model_pairs=12, fits=24, optimizer_updates=144)))

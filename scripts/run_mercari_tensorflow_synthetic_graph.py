"""Connect the declared TensorFlow feature/model/blend recipe on synthetic roles.

Current-runtime reconstruction with documented compatibility changes; not a
claim to reproduce the complete historical winning ensemble.
"""
# This import must precede sklearn/pandas/Arrow initialization on this host.
from probe_mercari_tensorflow_runtime import load_model_namespace
from probe_mercari_feature_runtime import load_feature_namespace
from mercari_checkpoint_probe import save_and_replay

import argparse
import ast
import faulthandler
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from joblib.externals import cloudpickle
from sciona.positive_log_blend import fit_positive_log_blend
from sciona.atoms.ml import positive_log_blending as blending_provider

ROOT = Path(__file__).resolve().parents[1]


def population(count, seed):
    rng = np.random.default_rng(seed)
    size = rng.integers(1, 9, count)
    family = rng.integers(0, 4, count)
    frames = pd.DataFrame([dict(name=f'synthetic sensor {s} cm',
        item_description=f'synthetic measurement {s} cm stable signal family {f}',
        category_name=f'synthetic/devices/group{f}', brand_name=f'maker{f}',
        shipping=int(f%2), item_condition_id=int(s%3+1)) for s, f in zip(size, family)])
    targets = 5. + size * 1.5 + family * .7 + rng.uniform(0, .3, count)
    return frames, targets


def run(directory, runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    if any(runtime.iterdir()):
        raise ValueError('Use an empty private runtime directory; existing checkpoints will not be replaced')
    model_ns, models, quantile_sha = load_model_namespace(directory)
    feature_ns, topology = load_feature_namespace(directory, modernize=True)
    # Source cache wrappers reference a transient exec namespace and cannot be
    # serialized. Preserve their pure functions, without process-local caches.
    for name in ('stem_word', 'word_to_charset'):
        feature_ns[name] = feature_ns[name].__wrapped__
    tree = ast.parse((directory/'mercari/datasets_tf.py').read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]),
                 '<source-feature-recipes>', 'exec'), feature_ns)
    triage = json.loads((ROOT/'docs/reviews/competition_mercari_source_triage.json').read_text())
    pins = {r['path']: r['sha256'] for r in triage['source_files']}
    for name in ('mercari/main_helpers.py', 'mercari/config.py'):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != pins[name]:
            raise ValueError('Pinned blend source changed: ' + name)
    config = ast.parse((directory/'mercari/config.py').read_text())
    bounds = {t.id: ast.literal_eval(n.value) for n in config.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ('MIN_PRICE_PRED', 'MAX_PRICE_PRED')}
    helper_tree = ast.parse((directory/'mercari/main_helpers.py').read_text())
    merge = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'merge_predictions')
    estimators = []
    def capture_lasso(*args, **kwargs):
        estimator = Lasso(*args, **kwargs)
        estimators.append(estimator)
        return estimator
    blend_ns = dict(np=np, Lasso=capture_lasso)
    exec(compile(ast.Module(body=[merge], type_ignores=[]), '<source-blend>', 'exec'), blend_ns)
    training, training_targets = population(8192, 857)
    calibration, calibration_targets = population(256, 859)
    query, _ = population(32, 863)
    calibration_predictions, query_predictions = {}, {}
    fits, rounds = [], []
    for group in (1, 3, 2):
        features = feature_ns[f'prepare_vectorizer_{group}_tf'](n_jobs=1)
        X = features.fit_transform(training.copy(deep=True), training_targets)
        with (runtime/f'features-{group}.pkl').open('wb') as stream:
            cloudpickle.dump(features, stream)
        C = features.transform(calibration.copy(deep=True))
        Q = features.transform(query.copy(deep=True))
        assert X.shape[1] == C.shape[1] == Q.shape[1]
        group_c, group_q = [], []
        for declared in [m for m in models['variants']['tensorflow'] if m['group'] == group]:
            controls = dict(declared['resolved_constructor_controls'])
            controls['actfunc'] = model_ns['prelu']
            model = model_ns[declared['model_class']](**controls)
            try:
                faulthandler.dump_traceback_later(120, exit=True)
                model.fit(X, np.log1p(training_targets), use_gpu=False, verbose=False)
                c, q = np.expm1(model.predict(C)), np.expm1(model.predict(Q))
                assert np.isfinite(c).all() and np.isfinite(q).all()
                np.testing.assert_array_equal(q, np.expm1(model.predict(Q)))
                expected_updates = sum((len(training) + controls['batch_size'] * 2 ** min(i, 2) - 1)
                                       // (controls['batch_size'] * 2 ** min(i, 2)) for i in range(3))
                assert model.update_count_ == expected_updates
                replay_c, replay_q = save_and_replay(model, model_ns[declared['model_class']], controls,
                    model_ns['tf'], runtime/f"model-{declared['seed']}", [C, Q])
                np.testing.assert_array_equal(replay_c, c)
                np.testing.assert_array_equal(replay_q, q)
                c, q = replay_c, replay_q
                group_c.append(c)
                group_q.append(q)
                fits.append(dict(group=group, seed=declared['seed'], epochs=3, optimizer_updates=model.update_count_,
                                 fresh_graph_checkpoint_replay_exact=True))
            finally:
                faulthandler.cancel_dump_traceback_later()
                if hasattr(model, '_session'):
                    model._session.close()
        calibration_predictions[group] = np.column_stack(group_c)
        query_predictions[group] = np.column_stack(group_q)
        # Source loads available saved groups in numeric order, regardless of
        # the launch order; this preserves the blend feature ordering.
        available = sorted(calibration_predictions)
        cp = np.hstack([calibration_predictions[g] for g in available]).clip(bounds['MIN_PRICE_PRED'], bounds['MAX_PRICE_PRED'])
        qp = np.hstack([query_predictions[g] for g in available]).clip(bounds['MIN_PRICE_PRED'], bounds['MAX_PRICE_PRED'])
        calibrated, predicted = blend_ns['merge_predictions'](cp, calibration_targets, qp, verbose=False)
        estimator = estimators[-1]
        reusable_blend = fit_positive_log_blend(cp, calibration_targets,
            clip_bounds=(bounds['MIN_PRICE_PRED'], bounds['MAX_PRICE_PRED']))
        np.testing.assert_allclose(reusable_blend.predict(qp), predicted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(reusable_blend.coefficients, estimator.coef_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(reusable_blend.intercept, estimator.intercept_, rtol=1e-12, atol=1e-12)
        column_ids = [f'group-{g}-model-{i}' for g in available for i in range(4)]
        provider_state = blending_provider.fit_positive_log_blend(cp, calibration_targets,
            column_ids, 'synthetic-response-coordinate.v1',
            [bounds['MIN_PRICE_PRED'], bounds['MAX_PRICE_PRED']],
            dict(alpha=.0001, max_iter=1000, tolerance=.0001, seed=9999))
        provider_state = json.loads(json.dumps(provider_state, allow_nan=False))
        provider_predictions = blending_provider.apply_positive_log_blend(qp, column_ids,
            'synthetic-response-coordinate.v1', provider_state)
        np.testing.assert_array_equal(provider_predictions, predicted)
        oracle = np.expm1(estimator.intercept_ + np.sum(np.log1p(qp) * estimator.coef_, axis=1))
        np.testing.assert_allclose(predicted, oracle, rtol=1e-12, atol=1e-12)
        assert predicted.shape == (32,) and np.isfinite(predicted).all()
        assert np.isfinite(calibrated).all() and (estimator.coef_ >= 0).all()
        rounds.append(dict(after_group=group, blend_group_order=available, blend_columns=cp.shape[1],
                           query_count=32, finite_predictions=True, independent_affine_oracle_passed=True,
                           reusable_blend_source_parity=True, registered_provider_json_parity=True))
        print(json.dumps(dict(completed_fits=len(fits), completed_blends=len(rounds))), flush=True)
    assert len(fits) == 12 and sum(f['optimizer_updates'] for f in fits) == 72
    with (runtime/'blend.pkl').open('wb') as stream:
        pickle.dump(estimators[-1], stream)
    with (runtime/'reusable-blend.pkl').open('wb') as stream:
        pickle.dump(reusable_blend, stream)
    (runtime/'provider-blend.json').write_text(json.dumps(provider_state, allow_nan=False)+'\n')
    np.save(runtime/'expected.npy', predicted, allow_pickle=False)
    manifest = dict(format='mercari-private-synthetic-replay.v1',
                    files={str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(runtime.rglob('*')) if p.is_file()})
    manifest_bytes = (json.dumps(manifest, sort_keys=True)+'\n').encode()
    (runtime/'manifest.json').write_bytes(manifest_bytes)
    child = subprocess.run([sys.executable, str(ROOT/'scripts/replay_mercari_synthetic_graph.py'),
        '--source-directory', str(directory), '--runtime-directory', str(runtime),
        '--manifest-sha256', hashlib.sha256(manifest_bytes).hexdigest()],
        env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
             'VECLIB_MAXIMUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1',
             'TF_NUM_INTRAOP_THREADS':'1', 'TF_NUM_INTEROP_THREADS':'1'},
        capture_output=True, text=True, timeout=120)
    if child.returncode:
        raise RuntimeError('Fresh-process replay failed: '+child.stderr[-2000:])
    replay_summary = json.loads(child.stdout.strip().splitlines()[-1])
    assert replay_summary == dict(passed=True, restored_models=12, restored_features=3, restored_blends=3,
                                  exact_query_predictions=True, training_executed=False)
    files = ['scripts/run_mercari_tensorflow_synthetic_graph.py', 'scripts/probe_mercari_tensorflow_runtime.py',
             'scripts/probe_mercari_feature_runtime.py', 'scripts/inventory_mercari_models.py', 'scripts/inventory_mercari_features.py',
             'scripts/mercari_checkpoint_probe.py', 'scripts/replay_mercari_synthetic_graph.py',
             'sciona/positive_log_blend.py', 'tests/test_positive_log_blend.py']
    provider_path = Path(blending_provider.__file__).resolve()
    provider_revision = subprocess.check_output(['git', '-C', str(provider_path.parents[4]), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(format='mercari-declared-tensorflow-synthetic-graph.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, role_counts=dict(training=8192, calibration=256, query=32),
        fits=fits, blend_rounds=rounds, optimizer_updates=72, checkpoint_replays=12,
        fresh_process_replay=replay_summary,
        source_commit=models['source_commit'],
        source_sha256={**models['source_sha256'], **topology['source_sha256'],
            'mercari/regression_clf.py': quantile_sha, 'mercari/main_helpers.py': pins['mercari/main_helpers.py'],
            'mercari/config.py': pins['mercari/config.py']},
        software_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        registered_blend_provider=dict(module='sciona.atoms.ml.positive_log_blending',
            revision=provider_revision, sha256=hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            functions=['fit_positive_log_blend', 'apply_positive_log_blend'], portable_json_state=True),
        runtime_versions={p: version(p) for p in ('tensorflow', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'Unidecode', 'nltk', 'joblib')},
        resource_policy=dict(concurrent_fits=1, native_pool_threads=1, worker_processes=0),
        limitations=['Declared 12-model TensorFlow path only; additional four winning models and backend identity remain unresolved.',
            'Current TensorFlow compatibility API and documented serial feature shims, not historical runtime numerical equivalence.',
            'Distinct synthetic training/calibration/query populations; no real-data accuracy or held-out model selection claim.',
            'Source MSE dispatch, feature recipes, clip bounds and learned positive Lasso blend retained, including group execution order 1,3,2.',
            'Fitted features, all 12 model checkpoints, target-inversion state and final blend replayed in a separate process without retraining; serialized code is trusted local diagnostic material, not a portable public provider format.',
            'Source text memoization wrappers removed for serialization; their underlying functions and output semantics are retained, with no cache-performance claim.',
            'This executes source via diagnostic loaders; reusable production providers, dependency publication review, stored corrected CDG and promotion remain outstanding.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        args = parser.parse_args()
        result = run(args.source_directory, args.runtime_directory)
    (ROOT/'docs/reviews/competition_mercari_tensorflow_synthetic_graph.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, fits=len(result['fits']), optimizer_updates=result['optimizer_updates'], approved=False)))

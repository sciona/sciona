"""Connect the declared TensorFlow feature/model/blend recipe on synthetic roles.

Current-runtime reconstruction with documented compatibility changes; not a
claim to reproduce the complete historical winning ensemble.
"""
# This import must precede sklearn/pandas/Arrow initialization on this host.
from probe_mercari_tensorflow_runtime import load_model_namespace
from probe_mercari_feature_runtime import load_feature_namespace

import argparse
import ast
import faulthandler
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

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


def run(directory):
    model_ns, models, quantile_sha = load_model_namespace(directory)
    feature_ns, topology = load_feature_namespace(directory, modernize=True)
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
                group_c.append(c)
                group_q.append(q)
                fits.append(dict(group=group, seed=declared['seed'], epochs=3, optimizer_updates=model.update_count_))
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
        oracle = np.expm1(estimator.intercept_ + np.sum(np.log1p(qp) * estimator.coef_, axis=1))
        np.testing.assert_allclose(predicted, oracle, rtol=1e-12, atol=1e-12)
        assert predicted.shape == (32,) and np.isfinite(predicted).all()
        assert np.isfinite(calibrated).all() and (estimator.coef_ >= 0).all()
        rounds.append(dict(after_group=group, blend_group_order=available, blend_columns=cp.shape[1],
                           query_count=32, finite_predictions=True, independent_affine_oracle_passed=True))
        print(json.dumps(dict(completed_fits=len(fits), completed_blends=len(rounds))), flush=True)
    assert len(fits) == 12 and sum(f['optimizer_updates'] for f in fits) == 72
    files = ['scripts/run_mercari_tensorflow_synthetic_graph.py', 'scripts/probe_mercari_tensorflow_runtime.py',
             'scripts/probe_mercari_feature_runtime.py', 'scripts/inventory_mercari_models.py', 'scripts/inventory_mercari_features.py']
    return dict(format='mercari-declared-tensorflow-synthetic-graph.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, role_counts=dict(training=8192, calibration=256, query=32),
        fits=fits, blend_rounds=rounds, optimizer_updates=72,
        source_commit=models['source_commit'],
        source_sha256={**models['source_sha256'], **topology['source_sha256'],
            'mercari/regression_clf.py': quantile_sha, 'mercari/main_helpers.py': pins['mercari/main_helpers.py'],
            'mercari/config.py': pins['mercari/config.py']},
        software_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        runtime_versions={p: version(p) for p in ('tensorflow', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'Unidecode', 'nltk')},
        resource_policy=dict(concurrent_fits=1, native_pool_threads=1, worker_processes=0),
        limitations=['Declared 12-model TensorFlow path only; additional four winning models and backend identity remain unresolved.',
            'Current TensorFlow compatibility API and documented serial feature shims, not historical runtime numerical equivalence.',
            'Distinct synthetic training/calibration/query populations; no real-data accuracy or held-out model selection claim.',
            'Source MSE dispatch, feature recipes, clip bounds and learned positive Lasso blend retained, including group execution order 1,3,2.',
            'This executes source via diagnostic loaders; reusable production providers, persisted checkpoint replay, dependency publication review, stored corrected CDG and promotion remain outstanding.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_mercari_tensorflow_synthetic_graph.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, fits=len(result['fits']), optimizer_updates=result['optimizer_updates'], approved=False)))

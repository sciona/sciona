"""Connect registered clipping, sparse models and blending on synthetic roles."""
from probe_mercari_tensorflow_runtime import load_model_namespace
from probe_mercari_feature_runtime import load_feature_namespace
from run_mercari_tensorflow_synthetic_graph import population
from validate_sparse_clip_feature_integration import RegisteredClip
from mercari_checkpoint_probe import replay_checkpoint

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import warnings
import numpy as np
from scipy import sparse
from joblib.externals import cloudpickle
from sciona.atoms.ml import sparse_neural_prediction as neural
from sciona.atoms.ml import positive_log_blending as blending
from sciona.atoms.ml import fitted_sparse_clipping as clipping

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(directory, expected):
    if sha(directory/'manifest.json') != expected:
        raise ValueError('Trusted synthetic manifest changed')
    manifest = json.loads((directory/'manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        path = directory/relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError('Trusted synthetic artifact changed')
    return manifest


def run(source, reference, reference_manifest, runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    if any(runtime.iterdir()):
        raise ValueError('Fresh private synthetic runtime directory required')
    verify_bundle(reference, reference_manifest)
    files = ['scripts/run_mercari_registered_models.py', 'scripts/replay_mercari_registered_models.py',
             'scripts/probe_mercari_tensorflow_runtime.py', 'scripts/probe_mercari_feature_runtime.py',
             'scripts/run_mercari_tensorflow_synthetic_graph.py', 'scripts/validate_sparse_clip_feature_integration.py',
             'scripts/mercari_checkpoint_probe.py', 'scripts/inventory_mercari_models.py',
             'scripts/inventory_mercari_features.py', 'sciona/sparse_neural_regression.py',
             'sciona/sparse_neural_state.py', 'sciona/sparse_neural_process.py', 'sciona/sparse_neural_worker.py',
             'sciona/quantile_target_encoding.py', 'sciona/fitted_sparse_clip.py', 'sciona/positive_log_blend.py']
    before = {p: sha(ROOT/p) for p in files}
    providers = {m.__name__: sha(m.__file__) for m in (neural, blending, clipping)}
    namespace, models, quantile_sha = load_model_namespace(source)
    features_ns, topology = load_feature_namespace(source, modernize=True)
    for name in ('stem_word', 'word_to_charset'):
        features_ns[name] = features_ns[name].__wrapped__
    features_ns['SanitizeSparseMatrix'] = RegisteredClip
    tree = ast.parse((source/'mercari/datasets_tf.py').read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]),
                 '<pinned-feature-recipes>', 'exec'), features_ns)
    training, targets = population(8192, 857)
    calibration, calibration_targets = population(256, 859)
    query, _ = population(32, 863)
    cp, qp, cases, rounds = {}, {}, [], []
    model_specs = []
    for group in (1, 3, 2):
        feature_pipeline = features_ns[f'prepare_vectorizer_{group}_tf'](n_jobs=1)
        X = feature_pipeline.fit_transform(training.copy(deep=True), targets)
        C = feature_pipeline.transform(calibration.copy(deep=True))
        Q = feature_pipeline.transform(query.copy(deep=True))
        with (reference/f'features-{group}.pkl').open('rb') as stream:
            reference_features = pickle.load(stream)
        RC = reference_features.transform(calibration.copy(deep=True))
        RQ = reference_features.transform(query.copy(deep=True))
        for original, current in ((RC, C), (RQ, Q)):
            canonical = RegisteredClip.canonical(original)
            for part in ('indptr', 'indices', 'data'):
                np.testing.assert_array_equal(getattr(canonical, part), getattr(current, part))
        with (runtime/f'features-{group}.pkl').open('wb') as stream:
            cloudpickle.dump(feature_pipeline, stream)
        columns = [f'group-{group}-slot-{j}' for j in range(X.shape[1])]
        group_c, group_q = [], []
        for declared in [r for r in models['variants']['tensorflow'] if r['group'] == group]:
            controls = dict(declared['resolved_constructor_controls'])
            if controls.pop('actfunc') != {'source_expression': 'prelu'}:
                raise ValueError('Expected declared PReLU activation')
            controls['n_hidden'] = list(controls['n_hidden'])
            controls.update(activation='prelu', quantile_precision=160., classification_loss_multiplier=6.)
            mode = 'quantile' if declared['model_class'] == 'RegressionClf' else 'mse'
            state = neural.fit_sparse_neural(X, np.log1p(targets), columns,
                'synthetic-comparable-features.v1', 'synthetic-log-response.v1', mode, controls)
            (runtime/f"model-{declared['seed']}.json").write_text(json.dumps(state, allow_nan=False))
            predictions = np.expm1(neural.predict_sparse_neural(sparse.vstack([C, Q], format='csr'), columns,
                'synthetic-comparable-features.v1', 'synthetic-log-response.v1', state))
            c, q = predictions[:len(calibration)], predictions[len(calibration):]
            original_controls = dict(declared['resolved_constructor_controls'], actfunc=namespace['prelu'])
            expected_c, expected_q = replay_checkpoint(namespace[declared['model_class']], original_controls,
                namespace['tf'], reference/f"model-{declared['seed']}", [RC, RQ])
            np.testing.assert_array_equal(c, expected_c)
            np.testing.assert_array_equal(q, expected_q)
            group_c.append(c)
            group_q.append(q)
            model_specs.append(dict(group=group, seed=declared['seed']))
            cases.append(dict(group=group, seed=declared['seed'], mode=mode,
                              exact_source_calibration=True, exact_source_query=True,
                              full_worker_schedule_verified=True))
            print(json.dumps(dict(completed_registered_fits=len(cases))), flush=True)
        cp[group], qp[group] = np.column_stack(group_c), np.column_stack(group_q)
        available = sorted(cp)
        identities = [f'group-{g}-model-{i}' for g in available for i in range(4)]
        state = blending.fit_positive_log_blend(np.hstack([cp[g] for g in available]), calibration_targets,
            identities, 'synthetic-response-coordinate.v1', [3., 2000.],
            dict(alpha=.0001, max_iter=1000, tolerance=.0001, seed=9999))
        final = blending.apply_positive_log_blend(np.hstack([qp[g] for g in available]), identities,
                                                 'synthetic-response-coordinate.v1', state)
        rounds.append(dict(after_group=group, model_count=len(identities), finite=bool(np.isfinite(final).all())))
    if len(cases) != 12 or len(rounds) != 3 or not all(r['finite'] for r in rounds):
        raise ValueError('Complete connected recipe required')
    np.testing.assert_array_equal(final, np.load(reference/'expected.npy', allow_pickle=False))
    (runtime/'blend.json').write_text(json.dumps(state, allow_nan=False))
    np.save(runtime/'expected.npy', final, allow_pickle=False)
    manifest = dict(format='mercari-registered-synthetic.v1', models=sorted(model_specs, key=lambda r:r['seed']),
                    files={p.name: sha(p) for p in sorted(runtime.iterdir()) if p.is_file()})
    (runtime/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    child = subprocess.run([sys.executable, str(ROOT/'scripts/replay_mercari_registered_models.py'),
        '--runtime-directory', str(runtime), '--manifest-sha256', sha(runtime/'manifest.json')],
        capture_output=True, text=True, timeout=180)
    if child.returncode:
        raise ValueError('Registered graph replay failed: '+child.stderr[-1500:])
    replay = json.loads(child.stdout.strip().splitlines()[-1])
    if replay != dict(passed=True, restored_models=12, exact_final_predictions=True, training_executed=False):
        raise ValueError('Complete fresh-process registered replay required')
    verify_bundle(reference, reference_manifest)
    if before != {p: sha(ROOT/p) for p in files} or providers != {m.__name__: sha(m.__file__) for m in (neural, blending, clipping)}:
        raise ValueError('Implementation changed during connected run')
    return dict(format='mercari-registered-model-graph.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, fits=cases, blend_rounds=rounds, fresh_process_replay=replay,
        role_counts=dict(training=8192, calibration=256, query=32), exact_original_final_query=True,
        software_sha256=before, provider_sha256=providers,
        source_commit=models['source_commit'], source_sha256={**models['source_sha256'], **topology['source_sha256'],
            'mercari/regression_clf.py': quantile_sha},
        resource_policy=dict(concurrent_fits=1, numerical_worker_processes=1, native_pool_threads=1,
                             parent_waits_during_worker_compute=True),
        limitations=['Only declared twelve-model TensorFlow recipe; original winner backend and extra models remain unresolved.',
            'Feature pipelines still use pinned source through diagnostic compatibility loaders and trusted private pickle replay.',
            'Registered clipping, model fit/predict and blend operations are connected; no original CDG approval is implied.',
            'Feature slot identities bind these fitted pipelines, not independently fitted vocabularies.',
            'Current-runtime synthetic equality does not establish historical-runtime equivalence or new-domain accuracy.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--reference-manifest-sha256', required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run(args.source_directory, args.reference_directory, args.reference_manifest_sha256, args.runtime_directory)
    (ROOT/'docs/reviews/mercari_registered_model_graph.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, fits=12, blends=3, exact_original_final_query=True)))

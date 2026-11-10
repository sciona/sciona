"""Measure clipping-provider substitution through twelve saved model checkpoints.

Only trusted local synthetic bundles are supported. No fitting occurs, and
diagnostic feature/model loaders remain outside catalog publication scope.
"""
# TensorFlow must initialize before sklearn/pandas/Arrow on this host.
from probe_mercari_tensorflow_runtime import load_model_namespace
from mercari_checkpoint_probe import replay_checkpoint
from run_mercari_tensorflow_synthetic_graph import population
from validate_sparse_clip_feature_integration import RegisteredClip

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
from sciona.atoms.ml import fitted_sparse_clipping, positive_log_blending

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(left, right):
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Finite aligned prediction vectors required')
    return dict(exact=bool(np.array_equal(left, right)),
                differing_values=int(np.count_nonzero(left != right)),
                max_absolute_difference=float(np.max(np.abs(left - right))))


def validate(directory, runtime, manifest_sha):
    files = ['scripts/validate_sparse_clip_checkpoint_replay.py',
             'scripts/validate_sparse_clip_feature_integration.py',
             'scripts/probe_mercari_tensorflow_runtime.py', 'scripts/mercari_checkpoint_probe.py',
             'scripts/run_mercari_tensorflow_synthetic_graph.py', 'sciona/fitted_sparse_clip.py']
    before = {p: sha(ROOT/p) for p in files}
    providers = {p.__name__: sha(p.__file__) for p in (fitted_sparse_clipping, positive_log_blending)}
    raw = (runtime/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha:
        raise ValueError('Trusted local synthetic manifest changed')
    manifest = json.loads(raw)
    if manifest['format'] != 'mercari-private-synthetic-replay.v1':
        raise ValueError('Unknown replay bundle format')
    def verify_bundle():
        for relative, digest in manifest['files'].items():
            path = runtime/relative
            if path.is_symlink() or not path.resolve().is_relative_to(runtime.resolve()) or sha(path) != digest:
                raise ValueError('Private synthetic artifact integrity failed')
    verify_bundle()
    namespace, models, _ = load_model_namespace(directory)
    calibration, _ = population(256, 859)
    query, _ = population(32, 863)
    cases, reference_predictions, provider_predictions = [], [], []
    for group in (1, 2, 3):
        with (runtime/f'features-{group}.pkl').open('rb') as stream:
            reference = pickle.load(stream)
        with (runtime/f'features-{group}.pkl').open('rb') as stream:
            candidate = pickle.load(stream)
        C = reference.transform(calibration.copy(deep=True))
        Q = reference.transform(query.copy(deep=True))
        replacements = 0
        for i, (name, stage) in enumerate(candidate.steps):
            if type(stage).__name__ == 'SanitizeSparseMatrix':
                adapter = RegisteredClip()
                adapter.columns_ = [f'pipeline-slot-{j}' for j in range(Q.shape[1])]
                adapter.state_ = json.loads(json.dumps(dict(format='sciona.fitted_sparse_clip.v1',
                    feature_columns=adapter.columns_, feature_coordinate='synthetic-comparable-features.v1',
                    upper=float(stage.datamax)), allow_nan=False))
                candidate.steps[i] = (name, adapter)
                replacements += 1
        if replacements != 1:
            raise ValueError('Exactly one frozen clipping stage required')
        CC = candidate.transform(calibration.copy(deep=True))
        QQ = candidate.transform(query.copy(deep=True))
        for expected, actual in [(C, CC), (Q, QQ)]:
            normalized = RegisteredClip.canonical(expected)
            for part in ('indptr', 'indices', 'data'):
                np.testing.assert_array_equal(getattr(normalized, part), getattr(actual, part))
        for declared in [m for m in models['variants']['tensorflow'] if m['group'] == group]:
            controls = dict(declared['resolved_constructor_controls'], actfunc=namespace['prelu'])
            faulthandler.dump_traceback_later(120, exit=True)
            try:
                c, q, cc, qq = replay_checkpoint(namespace[declared['model_class']], controls,
                    namespace['tf'], runtime/f"model-{declared['seed']}", [C, Q, CC, QQ])
            finally:
                faulthandler.cancel_dump_traceback_later()
            cases.append(dict(group=group, seed=declared['seed'],
                              calibration=compare(c, cc), query=compare(q, qq)))
            reference_predictions.append(q)
            provider_predictions.append(qq)
    state = json.loads((runtime/'provider-blend.json').read_text())
    def blend(vectors):
        return positive_log_blending.apply_positive_log_blend(np.column_stack(vectors),
            state['model_columns'], 'synthetic-response-coordinate.v1', state)
    expected = np.load(runtime/'expected.npy', allow_pickle=False)
    baseline = blend(reference_predictions)
    np.testing.assert_array_equal(baseline, expected)
    final = compare(baseline, blend(provider_predictions))
    verify_bundle()
    if before != {p: sha(ROOT/p) for p in files} or providers != {
            p.__name__: sha(p.__file__) for p in (fitted_sparse_clipping, positive_log_blending)}:
        raise ValueError('Implementation changed during replay')
    return dict(format='sparse-clip-checkpoint-replay.v1', measurement_completed=True,
        exact_equivalence=all(c[k]['exact'] for c in cases for k in ('calibration', 'query')) and final['exact'],
        approved=False, catalog_mutations=0, synthetic_only=True, training_executed=False,
        restored_models=len(cases), cases=cases, final_registered_blend=final,
        original_query_replay_exact=True, software_sha256=before, provider_sha256=providers,
        source_commit=models['source_commit'], source_sha256=models['source_sha256'],
        resource_policy=dict(concurrent_sessions=1, native_threads=1, worker_processes=0),
        limitations=['Frozen clipping bounds are transferred from trusted synthetic fitted source state; this replay does not refit any stage.',
                    'Explicit index sorting may alter floating-point reduction order. Exact equality is measured separately for every model and final blend.',
                    'Diagnostic source feature/model loaders and the original competition CDG remain unapproved.',
                    'No real data, accuracy claim, historical-runtime equivalence or arbitrary pickle loading is qualified.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = validate(args.source_directory, args.runtime_directory, args.manifest_sha256)
    (ROOT/'docs/reviews/sparse_clip_checkpoint_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('measurement_completed', 'exact_equivalence', 'restored_models', 'final_registered_blend')}))

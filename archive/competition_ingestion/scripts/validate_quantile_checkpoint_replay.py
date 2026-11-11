"""Measure quantile-provider substitution through twelve saved model checkpoints.

Only trusted local synthetic bundles are supported. No neural training occurs;
diagnostic feature/model loaders remain outside catalog publication scope.
"""
# TensorFlow must initialize before sklearn/pandas/Arrow on this host.
from probe_mercari_tensorflow_runtime import load_model_namespace
from mercari_checkpoint_probe import replay_checkpoint
from run_mercari_tensorflow_synthetic_graph import population

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
from sciona.atoms.ml import quantile_response_encoding, positive_log_blending

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
    files = ['scripts/validate_quantile_checkpoint_replay.py',
             'scripts/probe_mercari_tensorflow_runtime.py', 'scripts/mercari_checkpoint_probe.py',
             'scripts/run_mercari_tensorflow_synthetic_graph.py', 'sciona/quantile_target_encoding.py']
    before = {p: sha(ROOT/p) for p in files}
    providers = {p.__name__: sha(p.__file__) for p in (quantile_response_encoding, positive_log_blending)}
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
    namespace, models, quantile_sha = load_model_namespace(directory)
    _, training_targets = population(8192, 857)
    log_targets = np.log1p(training_targets)
    calibration, _ = population(256, 859)
    query, _ = population(32, 863)
    cases, reference_predictions, provider_predictions = [], [], []
    for group in (1, 2, 3):
        with (runtime/f'features-{group}.pkl').open('rb') as stream:
            reference = pickle.load(stream)
        C = reference.transform(calibration.copy(deep=True))
        Q = reference.transform(query.copy(deep=True))
        for declared in [m for m in models['variants']['tensorflow'] if m['group'] == group]:
            controls = dict(declared['resolved_constructor_controls'], actfunc=namespace['prelu'])
            base_class = namespace[declared['model_class']]
            is_classifier = declared['model_class'] == 'RegressionClf'
            encoding_exact = None
            if is_classifier:
                fitted = quantile_response_encoding.fit_quantile_response(
                    log_targets, 'synthetic-log-response.v1', controls['n_bins'])
                source_model = base_class(**controls)
                expected_encoding = source_model._scale_target(log_targets)
                encoded = quantile_response_encoding.encode_quantile_response(
                    log_targets, 'synthetic-log-response.v1', fitted, True, 160.)
                np.testing.assert_array_equal(encoded, expected_encoding)
                with (runtime/f"model-{declared['seed']}"/'inverse-state.pkl').open('rb') as stream:
                    stored = pickle.load(stream)
                np.testing.assert_array_equal(fitted['boundaries'], stored['percentiles'])
                encoding_exact = True
                class ProviderModel(base_class):
                    def _invert_target(self, probabilities):
                        state = dict(format='sciona.quantile_response.v1',
                            response_coordinate='synthetic-log-response.v1', boundaries=self.percentiles.tolist())
                        state = json.loads(json.dumps(state, allow_nan=False))
                        return quantile_response_encoding.decode_quantile_response(
                            probabilities, 'synthetic-log-response.v1', state)
                candidate_class = ProviderModel
            else:
                candidate_class = base_class
            faulthandler.dump_traceback_later(120, exit=True)
            try:
                c, q = replay_checkpoint(base_class, controls,
                    namespace['tf'], runtime/f"model-{declared['seed']}", [C, Q])
                cc, qq = replay_checkpoint(candidate_class, controls,
                    namespace['tf'], runtime/f"model-{declared['seed']}", [C, Q])
            finally:
                faulthandler.cancel_dump_traceback_later()
            cases.append(dict(group=group, seed=declared['seed'], quantile_model=is_classifier,
                              full_training_encoding_exact=encoding_exact,
                              calibration=compare(c, cc), query=compare(q, qq)))
            reference_predictions.append(q)
            provider_predictions.append(qq)
    if len(cases) != 12 or sum(c['quantile_model'] for c in cases) != 6:
        raise ValueError('Expected twelve models including six quantile classifiers')
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
            p.__name__: sha(p.__file__) for p in (quantile_response_encoding, positive_log_blending)}:
        raise ValueError('Implementation changed during replay')
    return dict(format='quantile-checkpoint-replay.v1', measurement_completed=True,
        exact_equivalence=all(c[k]['exact'] for c in cases for k in ('calibration', 'query')) and final['exact'],
        approved=False, catalog_mutations=0, synthetic_only=True, neural_training_executed=False,
        quantile_fitting_executed=True,
        restored_models=len(cases), quantile_models=sum(c['quantile_model'] for c in cases), cases=cases, final_registered_blend=final,
        original_query_replay_exact=True, software_sha256=before, provider_sha256=providers,
        source_commit=models['source_commit'], source_sha256={**models['source_sha256'],
            'mercari/regression_clf.py': quantile_sha},
        resource_policy=dict(concurrent_sessions=1, native_threads=1, worker_processes=0),
        limitations=['Quantile state is refitted on the original synthetic training targets and compared exactly with saved boundaries; no neural model is retrained.',
                    'All synthetic training encodings and checkpoint calibration/query predictions are checked; fresh trained-model equivalence is not established here.',
                    'Non-classification models are unchanged controls. Diagnostic source feature/model loaders and the original competition CDG remain unapproved.',
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
    (ROOT/'docs/reviews/quantile_checkpoint_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('measurement_completed', 'exact_equivalence', 'restored_models', 'final_registered_blend')}))

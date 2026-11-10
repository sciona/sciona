"""Replay only the trusted synthetic bundle produced by the paired runner."""
from probe_mercari_tensorflow_runtime import load_model_namespace
from mercari_checkpoint_probe import replay_checkpoint
from run_mercari_tensorflow_synthetic_graph import population

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
from sciona.atoms.ml.positive_log_blending import apply_positive_log_blend


def replay(directory, runtime, expected_manifest_sha):
    raw = (runtime/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha:
        raise ValueError('Private replay manifest changed')
    manifest = json.loads(raw)
    if manifest['format'] != 'mercari-private-synthetic-replay.v1':
        raise ValueError('Unknown private replay format')
    for relative, digest in manifest['files'].items():
        path = runtime/relative
        if path.is_symlink() or not path.resolve().is_relative_to(runtime.resolve()):
            raise ValueError('Replay artifact escaped private bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Replay artifact changed')
    namespace, models, _ = load_model_namespace(directory)
    query, _ = population(32, 863)
    predictions = []
    for group in (1, 2, 3):
        with (runtime/f'features-{group}.pkl').open('rb') as stream:
            features = pickle.load(stream)
        Q = features.transform(query.copy(deep=True))
        for declared in [m for m in models['variants']['tensorflow'] if m['group'] == group]:
            controls = dict(declared['resolved_constructor_controls'])
            controls['actfunc'] = namespace['prelu']
            result, = replay_checkpoint(namespace[declared['model_class']], controls, namespace['tf'],
                                       runtime/f"model-{declared['seed']}", [Q])
            predictions.append(result)
    with (runtime/'blend.pkl').open('rb') as stream:
        blend = pickle.load(stream)
    # Bounds are fixed by the source config verified by the parent run.
    matrix = np.column_stack(predictions).clip(3, 2000)
    actual = np.expm1(blend.predict(np.log1p(matrix)))
    expected = np.load(runtime/'expected.npy', allow_pickle=False)
    np.testing.assert_array_equal(actual, expected)
    with (runtime/'reusable-blend.pkl').open('rb') as stream:
        reusable_blend = pickle.load(stream)
    np.testing.assert_array_equal(reusable_blend.predict(matrix), expected)
    provider_state = json.loads((runtime/'provider-blend.json').read_text())
    provider_predictions = apply_positive_log_blend(matrix, provider_state['model_columns'],
        'synthetic-response-coordinate.v1', provider_state)
    np.testing.assert_array_equal(provider_predictions, expected)
    return dict(passed=True, restored_models=12, restored_features=3, restored_blends=3,
                exact_query_predictions=True, training_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = replay(args.source_directory, args.runtime_directory, args.manifest_sha256)
    print(json.dumps(result))

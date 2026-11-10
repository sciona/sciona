"""Replay trusted synthetic features with numeric registered model states."""
from run_mercari_registered_models import verify_bundle
from run_mercari_tensorflow_synthetic_graph import population
import argparse
import json
from pathlib import Path
import pickle
import warnings
import numpy as np
from sciona.atoms.ml.sparse_neural_prediction import predict_sparse_neural
from sciona.atoms.ml.positive_log_blending import apply_positive_log_blend


def replay(runtime, digest):
    manifest = verify_bundle(runtime, digest)
    if manifest['format'] != 'mercari-registered-synthetic.v1' or len(manifest['models']) != 12:
        raise ValueError('Exact registered synthetic bundle required')
    query, _ = population(32, 863)
    matrices = {}
    predictions = []
    for item in manifest['models']:
        group = item['group']
        if group not in matrices:
            with (runtime/f'features-{group}.pkl').open('rb') as stream:
                features = pickle.load(stream)
            matrices[group] = features.transform(query.copy(deep=True))
        Q = matrices[group]
        state = json.loads((runtime/f"model-{item['seed']}.json").read_text())
        predictions.append(np.expm1(predict_sparse_neural(Q,
            [f'group-{group}-slot-{j}' for j in range(Q.shape[1])],
            'synthetic-comparable-features.v1', 'synthetic-log-response.v1', state)))
    state = json.loads((runtime/'blend.json').read_text())
    actual = apply_positive_log_blend(np.column_stack(predictions), state['model_columns'],
                                     'synthetic-response-coordinate.v1', state)
    np.testing.assert_array_equal(actual, np.load(runtime/'expected.npy', allow_pickle=False))
    return dict(passed=True, restored_models=12, exact_final_predictions=True, training_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(json.dumps(replay(args.runtime_directory, args.manifest_sha256)))

"""Replay a private synthetic graph using JSON states and known providers only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sciona.mercari_feature_state import restore_mercari_features
from sciona.mercari_table_preprocessing import MercariKLPreprocessor
from sciona.serial_feature_union import SerialFeatureUnion
from sciona.sparse_clip_transformer import SparseClipTransformer
from sciona.atoms.ml import sparse_neural_prediction as neural
from sciona.atoms.ml import positive_log_blending as blending


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, digest):
    if sha(directory/'manifest.json') != digest:
        raise ValueError('Synthetic manifest changed')
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = directory/name
        if path.is_symlink() or path.parent != directory or sha(path) != expected:
            raise ValueError('Synthetic bundle changed')
    return manifest


def forbidden(*args, **kwargs):
    raise AssertionError('Prediction replay attempted fitting')


def replay(directory, digest):
    manifest = verify(directory, digest)
    models = [dict(group=g, seed=g*100+i) for g in (1, 2, 3) for i in range(4)]
    if manifest['format'] != 'mercari-portable-synthetic.v1' or manifest['models'] != models:
        raise ValueError('Complete declared model ordering required')
    for cls in (DictVectorizer, CountVectorizer, TfidfVectorizer, MercariKLPreprocessor, SparseClipTransformer):
        cls.fit = cls.fit_transform = forbidden
    SerialFeatureUnion.fit_transform = forbidden
    neural.fit_sparse_neural = blending.fit_positive_log_blend = forbidden
    query = pd.DataFrame(json.loads((directory/'query.json').read_text()))
    matrices, predictions = {}, []
    for item in models:
        group = item['group']
        if group not in matrices:
            state = json.loads((directory/f'group-{group}.json').read_text())
            matrices[group] = restore_mercari_features(state).transform(query)
        matrix = matrices[group]
        state = json.loads((directory/f"model-{item['seed']}.json").read_text())
        predictions.append(np.expm1(neural.predict_sparse_neural(matrix,
            [f'group-{group}-slot-{i}' for i in range(matrix.shape[1])],
            'synthetic-comparable-features.v1', 'synthetic-log-response.v1', state)))
    blend = json.loads((directory/'blend.json').read_text())
    columns = [f'group-{g}-model-{i}' for g in (1, 2, 3) for i in range(4)]
    actual = blending.apply_positive_log_blend(np.column_stack(predictions), columns,
                                              'synthetic-response-coordinate.v1', blend)
    np.testing.assert_array_equal(actual, np.load(directory/'expected.npy', allow_pickle=False))
    verify(directory, digest)
    return dict(passed=True, restored_feature_pipelines=3, restored_models=12,
                restored_blends=1, exact_final_predictions=True, training_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.directory, args.manifest_sha256)))

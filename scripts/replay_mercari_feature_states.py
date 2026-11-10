"""Replay private synthetic feature packages without fitting or source loaders."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sciona.mercari_feature_state import restore_mercari_features
from sciona.serial_feature_union import SerialFeatureUnion
from sciona.mercari_table_preprocessing import MercariKLPreprocessor
from sciona.sparse_clip_transformer import SparseClipTransformer


def forbidden(*args, **kwargs):
    raise AssertionError('Prediction replay attempted fitting')


def replay(directory):
    for cls in (DictVectorizer, CountVectorizer, TfidfVectorizer, MercariKLPreprocessor, SparseClipTransformer):
        cls.fit = forbidden
        cls.fit_transform = forbidden
    SerialFeatureUnion.fit_transform = forbidden
    cases = 0
    for group in (1, 3, 2):
        model = restore_mercari_features(json.loads((directory/f'group-{group}.json').read_text()))
        for role in ('calibration', 'query'):
            frame = pd.DataFrame(json.loads((directory/f'{role}.json').read_text()))
            actual = model.transform(frame)
            expected = load_npz(directory/f'group-{group}-{role}.npz')
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            for part in ('data', 'indices', 'indptr'):
                np.testing.assert_array_equal(getattr(actual, part), getattr(expected, part))
            cases += 1
    return dict(passed=True, restored_pipelines=3, exact_heldout_matrices=cases, training_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(replay(parser.parse_args().directory)))

"""Whole-feature prediction state for the explicit Mercari application adapter.

Only known six-stage feature pipelines are accepted. All state is JSON data;
private learned vocabularies and category names must remain runtime inputs.
This packaging does not approve the original CDG, its dependencies or any
historical winning ensemble. Restored feature unions cannot be refitted.
"""
import json
from importlib.metadata import version
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sciona.mercari_table_preprocessing import MercariFillEmpty, MercariKLPreprocessor, export_kl_preprocessor, restore_kl_preprocessor
from sciona.mercari_pj_preprocessing import MercariPJPreprocessor, export_pj_preprocessor, restore_pj_preprocessor
from sciona.serial_feature_union import SerialFeatureUnion
from sciona.table_feature_operations import NumericCSR
from sciona.feature_transform_state import export_feature_transform, restore_feature_transform
from sciona.sparse_clip_transformer import SparseClipTransformer
from sciona.atoms.ml.fitted_sparse_clipping import apply_sparse_clip


def _runtime():
    return {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'nltk', 'Unidecode')}


def export_mercari_features(model):
    if (type(model) is not Pipeline or len(model.steps) != 6 or model.memory is not None
            or model.transform_input is not None or model.verbose):
        raise ValueError('Known six-stage application pipeline required')
    fill, policy, branches, cast, clip, reporter = [stage for _, stage in model.steps]
    if (type(fill) is not MercariFillEmpty or type(branches) is not SerialFeatureUnion
            or type(cast) is not NumericCSR or cast.dtype != 'float32'
            or type(clip) is not SparseClipTransformer or clip.state_ is None
            or type(reporter) is not FunctionTransformer
            or reporter.get_params() != FunctionTransformer().get_params()):
        raise ValueError('Known feature stages required')
    if type(policy) is MercariKLPreprocessor:
        preprocessing = dict(kind='kl', state=export_kl_preprocessor(policy))
    elif type(policy) is MercariPJPreprocessor:
        preprocessing = dict(kind='pj', state=export_pj_preprocessor(policy))
    else:
        raise ValueError('Explicit known application preprocessing required')
    state = dict(format='sciona.mercari_features_prediction.v1', runtime=_runtime(),
        step_names=[name for name, _ in model.steps], preprocessing=preprocessing,
        branches=export_feature_transform(branches), feature_coordinate=clip.feature_coordinate,
        clipping=clip.state_)
    state = json.loads(json.dumps(state, allow_nan=False))
    # Validate the complete package before releasing it; no fitting occurs.
    restore_mercari_features(state)
    return state


def restore_mercari_features(state):
    fields = {'format', 'runtime', 'step_names', 'preprocessing', 'branches', 'feature_coordinate', 'clipping'}
    if (type(state) is not dict or set(state) != fields
            or state['format'] != 'sciona.mercari_features_prediction.v1' or state['runtime'] != _runtime()):
        raise ValueError('Known exact-runtime application feature state required')
    state = json.loads(json.dumps(state, allow_nan=False))
    names = state['step_names']
    if (type(names) is not list or len(names) != 6
            or any(type(n) is not str or not n.strip() or '__' in n for n in names)
            or len(set(names)) != 6):
        raise ValueError('Six unique explicit stage names required')
    policy = state['preprocessing']
    if type(policy) is not dict or set(policy) != {'kind', 'state'}:
        raise ValueError('Exact preprocessing policy required')
    if policy['kind'] == 'kl':
        preprocessing = restore_kl_preprocessor(policy['state'])
    elif policy['kind'] == 'pj':
        preprocessing = restore_pj_preprocessor(policy['state'])
    else:
        raise ValueError('Unknown application policy')
    branches = restore_feature_transform(state['branches'])
    if type(branches) is not SerialFeatureUnion:
        raise ValueError('Serial feature branches required')
    width = sum(row[3] for row in branches._fitted)
    columns = [f'pipeline-slot-{i}' for i in range(width)]
    # Validate width, coordinate, clipping schema and bounds through the provider.
    apply_sparse_clip(csr_matrix((0, width), dtype=np.float32), columns,
                      state['feature_coordinate'], state['clipping'])
    clip = SparseClipTransformer(state['feature_coordinate'])
    clip.state_ = state['clipping']
    stages = [MercariFillEmpty(), preprocessing, branches, NumericCSR('float32'), clip, FunctionTransformer()]
    result = Pipeline(list(zip(names, stages)))
    result._validate_steps()
    return result

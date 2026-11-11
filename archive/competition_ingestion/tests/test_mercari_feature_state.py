import json
import numpy as np
import pandas as pd
import pytest
from sciona.mercari_feature_recipes import prepare_vectorizer_2_tf
from sciona.mercari_feature_state import export_mercari_features, restore_mercari_features


@pytest.fixture
def fitted():
    frame = pd.DataFrame(dict(name=['alpha sensor', 'beta sensor', 'gamma reactor', 'delta reactor'],
        item_description=['synthetic measurement stable', 'synthetic signal stable', 'synthetic reaction stable', 'synthetic device stable'],
        category_name=['synthetic/devices'] * 4, brand_name=['maker'] * 4, shipping=[0, 1, 0, 1], item_condition_id=[1, 2, 1, 2]))
    model = prepare_vectorizer_2_tf(feature_coordinate='synthetic-space')
    model.fit_transform(frame)
    return model, frame


def test_full_json_roundtrip(fitted):
    model, frame = fitted
    state = json.loads(json.dumps(export_mercari_features(model), allow_nan=False))
    restored = restore_mercari_features(state)
    expected, actual = model.transform(frame), restored.transform(frame)
    for key in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert export_mercari_features(restored) == state


@pytest.mark.parametrize('fault', ['coordinate', 'width', 'names', 'policy', 'runtime', 'extra'])
def test_invalid_complete_state_rejected(fitted, fault):
    model, frame = fitted
    state = export_mercari_features(model)
    if fault == 'coordinate': state['feature_coordinate'] = 'different'
    if fault == 'width': state['branches']['payload'][0]['width'] += 1
    if fault == 'names': state['step_names'][1] = state['step_names'][0]
    if fault == 'policy': state['preprocessing']['kind'] = 'python'
    if fault == 'runtime': state['runtime'] = {}
    if fault == 'extra': state['executable'] = 'untrusted'
    with pytest.raises(ValueError): restore_mercari_features(state)

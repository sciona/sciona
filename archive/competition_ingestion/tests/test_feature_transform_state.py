import copy
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sciona.table_feature_operations import TableColumnSelector
from sciona.serial_feature_union import SerialFeatureUnion
from sciona.feature_transform_state import export_feature_transform, restore_feature_transform


def fitted():
    frame = pd.DataFrame({'left': ['alpha beta', 'gamma alpha'], 'right': ['delta beta', 'delta epsilon']})
    branches = [(name, [column], Pipeline([('select', TableColumnSelector([column])),
        ('encode', CountVectorizer(dtype=np.float32)), ('identity', FunctionTransformer())]))
        for name, column in [('first', 'right'), ('second', 'left')]]
    union = SerialFeatureUnion(branches)
    union.fit_transform(frame)
    return union, frame


def test_composed_json_replay_and_refit_rejected():
    union, frame = fitted()
    state = json.loads(json.dumps(export_feature_transform(union)))
    restored = restore_feature_transform(state)
    expected, actual = union.transform(frame), restored.transform(frame)
    for key in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert export_feature_transform(restored) == state
    with pytest.raises(ValueError): restored.fit_transform(frame)


def test_unknown_and_executable_operations_rejected():
    with pytest.raises(ValueError): export_feature_transform(FunctionTransformer(lambda x: x))
    with pytest.raises(ValueError): export_feature_transform(object())
    state = export_feature_transform(FunctionTransformer())
    state['kind'] = 'import'
    with pytest.raises(ValueError): restore_feature_transform(state)


@pytest.mark.parametrize('fault', ['duplicate_name', 'columns', 'width', 'extra'])
def test_invalid_composition_rejected(fault):
    union, frame = fitted()
    state = copy.deepcopy(export_feature_transform(union))
    if fault == 'duplicate_name': state['payload'][1]['name'] = state['payload'][0]['name']
    if fault == 'columns': state['payload'][0]['columns'] = ['a', 'a']
    if fault == 'width': state['payload'][0]['width'] = True
    if fault == 'extra': state['payload'][0]['callable'] = 'module.function'
    with pytest.raises(ValueError): restore_feature_transform(state)


def test_fresh_process_composition_without_training():
    union, frame = fitted()
    payload = dict(state=export_feature_transform(union), records=frame.to_dict('records'))
    code = '''
import json, sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sciona.serial_feature_union import SerialFeatureUnion
from sciona.feature_transform_state import restore_feature_transform

def forbidden(*a, **k):
    raise AssertionError('Prediction restoration attempted fitting')
CountVectorizer.fit = forbidden
CountVectorizer.fit_transform = forbidden
SerialFeatureUnion.fit_transform = forbidden
payload = json.load(sys.stdin)
result = restore_feature_transform(payload['state']).transform(pd.DataFrame(payload['records']))
print(json.dumps(result.toarray().tolist()))
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', code], input=json.dumps(payload), env=env,
                            text=True, capture_output=True, check=True, timeout=60)
    assert json.loads(result.stdout) == union.transform(frame).toarray().tolist()

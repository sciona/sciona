import json
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer
from sciona.dict_vectorizer_state import export_dict_vectorizer, restore_dict_vectorizer

TRAIN = [{'kind': 'sensor', 'value': 1.5, 'tags': ['a', 'a']},
         {'kind': 'reactor', 'value': -2., 'tags': ['b']}]
QUERY = [{'kind': 'unknown', 'value': 3., 'tags': ['a', 'unknown']}, {},
         {'kind': 'sensor', 'value': 0.}]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('sparse', [True, False])
@pytest.mark.parametrize('sort', [True, False])
def test_exact_json_replay_without_fit(dtype, sparse, sort, monkeypatch):
    model = DictVectorizer(dtype=dtype, sparse=sparse, sort=sort, separator='::').fit(TRAIN)
    expected = model.transform(QUERY)
    state = json.loads(json.dumps(export_dict_vectorizer(model)))
    monkeypatch.setattr(DictVectorizer, 'fit', lambda *a, **k: pytest.fail('Replay fitted'))
    monkeypatch.setattr(DictVectorizer, 'fit_transform', lambda *a, **k: pytest.fail('Replay fitted'))
    restored = restore_dict_vectorizer(state)
    assert restored.feature_names_ == model.feature_names_
    actual = restored.transform(QUERY)
    assert actual.dtype == expected.dtype
    if sparse:
        for field in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    else:
        np.testing.assert_array_equal(actual, expected)
    state['feature_names'].reverse()
    assert restored.feature_names_ == model.feature_names_


@pytest.mark.parametrize('field,value', [('sklearn_version', 'unknown'), ('dtype', 'object'),
    ('feature_names', ['a', 'a']), ('feature_names', [1]), ('feature_names', []),
    ('sparse', 1), ('sort', 'yes'), ('separator', None), ('extra', 'ignored')])
def test_invalid_state_rejected(field, value):
    state = export_dict_vectorizer(DictVectorizer().fit(TRAIN))
    state[field] = value
    with pytest.raises(ValueError):
        restore_dict_vectorizer(state)


def test_inconsistent_fitted_order_rejected():
    model = DictVectorizer().fit(TRAIN)
    model.feature_names_.reverse()
    with pytest.raises(ValueError):
        export_dict_vectorizer(model)


def test_fresh_process_replay():
    model = DictVectorizer(dtype=np.float32).fit(TRAIN)
    payload = dict(state=export_dict_vectorizer(model), query=QUERY)
    code = '''
import json, sys
from sklearn.feature_extraction import DictVectorizer
from sciona.dict_vectorizer_state import restore_dict_vectorizer

def forbidden(*args, **kwargs):
    raise AssertionError('Prediction restoration must not train')
DictVectorizer.fit = forbidden
DictVectorizer.fit_transform = forbidden
payload = json.load(sys.stdin)
print(json.dumps(restore_dict_vectorizer(payload['state']).transform(payload['query']).toarray().tolist()))
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', code], input=json.dumps(payload), capture_output=True,
                            text=True, env=env, timeout=60, check=True)
    assert json.loads(result.stdout) == model.transform(QUERY).toarray().tolist()

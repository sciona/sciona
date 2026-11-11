import copy
import json

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sciona.character_run_tokenizer import CharacterRunTokenizer
from sciona.text_vectorizer_state import export_text_vectorizer, restore_text_vectorizer

TRAIN = ['alpha beta beta', 'beta gamma', 'alpha delta', 'gamma delta alpha']
QUERY = ['unknown alpha beta', 'gamma gamma', '']


@pytest.mark.parametrize('cls', [CountVectorizer, TfidfVectorizer])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('analyzer', ['word', 'char', 'char_wb'])
def test_json_replay_exact_and_never_refits(cls, dtype, analyzer, monkeypatch):
    model = cls(dtype=dtype, analyzer=analyzer, ngram_range=(1, 2), min_df=1)
    model.fit(TRAIN)
    expected = model.transform(QUERY)
    state = json.loads(json.dumps(export_text_vectorizer(model), allow_nan=False))
    monkeypatch.setattr(cls, 'fit', lambda *args, **kwargs: pytest.fail('Prediction restoration fitted'))
    monkeypatch.setattr(cls, 'fit_transform', lambda *args, **kwargs: pytest.fail('Prediction restoration fitted'))
    actual = restore_text_vectorizer(state).transform(QUERY)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    for field in ('indices', 'indptr', 'data'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_explicit_character_policy_and_no_idf(monkeypatch):
    model = TfidfVectorizer(dtype=np.float32, tokenizer=CharacterRunTokenizer('ACGT', ' '),
        lowercase=False, token_pattern=None, use_idf=False, norm='l1', sublinear_tf=True)
    model.fit(['ACG?TT', 'ACG!!', 'TT ACG'])
    state = json.loads(json.dumps(export_text_vectorizer(model)))
    monkeypatch.setattr(TfidfVectorizer, 'fit', lambda *a, **k: pytest.fail('Restoration fitted'))
    monkeypatch.setattr(TfidfVectorizer, 'fit_transform', lambda *a, **k: pytest.fail('Restoration fitted'))
    restored = restore_text_vectorizer(state)
    np.testing.assert_array_equal(restored.transform(['ACG??']).toarray(), model.transform(['ACG??']).toarray())


@pytest.mark.parametrize('field,value', [
    ('sklearn_version', 'unknown'), ('vocabulary', ['same', 'same']),
    ('idf', [float('nan')]), ('format', 'unknown'), ('mode', 'pickle'),
])
def test_corrupt_state_rejected(field, value):
    model = TfidfVectorizer(dtype=np.float32).fit(TRAIN)
    state = export_text_vectorizer(model)
    state[field] = value
    with pytest.raises(ValueError):
        restore_text_vectorizer(state)


def test_executable_and_unknown_parameters_rejected():
    model = CountVectorizer(dtype=np.float32).fit(TRAIN)
    original = export_text_vectorizer(model)
    for key, value in [('preprocessor', 'some.module.function'), ('tokenizer', {'kind': 'python'}),
                       ('extra', True), ('analyzer', 'callable')]:
        state = copy.deepcopy(original)
        state['parameters'][key] = value
        with pytest.raises(ValueError):
            restore_text_vectorizer(state)
    model.preprocessor = lambda text: text
    with pytest.raises(ValueError):
        export_text_vectorizer(model)


def test_fresh_process_json_replay_without_training():
    import os
    import subprocess
    import sys
    models = [CountVectorizer(dtype=np.float32), TfidfVectorizer(dtype=np.float64),
              TfidfVectorizer(dtype=np.float32, use_idf=False)]
    payload = []
    expected = []
    for model in models:
        model.fit(TRAIN)
        payload.append(export_text_vectorizer(model))
        expected.append(model.transform(QUERY).toarray().tolist())
    code = '''
import json, sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sciona.text_vectorizer_state import restore_text_vectorizer

def forbidden(*args, **kwargs):
    raise AssertionError('Restoration must not train')
for cls in (CountVectorizer, TfidfVectorizer, TfidfTransformer):
    cls.fit = forbidden
    cls.fit_transform = forbidden
payload = json.load(sys.stdin)
print(json.dumps([restore_text_vectorizer(s).transform(payload['query']).toarray().tolist()
                  for s in payload['states']], allow_nan=False))
'''
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', code], input=json.dumps(dict(states=payload, query=QUERY)),
                            env=environment, capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(result.stdout) == expected

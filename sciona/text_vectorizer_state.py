"""Portable prediction state for known sklearn count and TF-IDF text encoders.

Vocabulary and IDF values are private fitted runtime inputs. State contains no
documents or executable objects. Compatibility is restricted to the exact
installed sklearn version; callers retain responsibility for text semantics,
training-only fitting and identical preprocessing at inference.
"""
import json
from importlib.metadata import version

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from sciona.character_run_tokenizer import CharacterRunTokenizer


def _parameters(model):
    params = model.get_params(deep=False)
    if (params['input'] != 'content' or params['preprocessor'] is not None
            or params['analyzer'] not in ('word', 'char', 'char_wb')):
        raise ValueError('Known content analyzer without executable preprocessing required')
    if params['strip_accents'] not in (None, 'ascii', 'unicode'):
        raise ValueError('Known accent policy required')
    if params['stop_words'] is not None:
        raise ValueError('Explicit upstream stop-word policy required')
    tokenizer = params['tokenizer']
    if tokenizer is not None and type(tokenizer) is not CharacterRunTokenizer:
        raise ValueError('Only explicit character-run tokenizer state is supported')
    params['tokenizer'] = None if tokenizer is None else dict(
        kind='character_runs', word_characters=tokenizer.word_characters,
        whitespace_characters=tokenizer.whitespace_characters)
    dtype = np.dtype(params['dtype'])
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Float32 or float64 text features required')
    params['dtype'] = dtype.name
    params['ngram_range'] = list(params['ngram_range'])
    # Vocabulary has one canonical ordered representation outside parameters.
    params['vocabulary'] = None
    json.dumps(params, allow_nan=False)
    return params


def export_text_vectorizer(model):
    if type(model) not in (CountVectorizer, TfidfVectorizer):
        raise ValueError('Known count or TF-IDF encoder required')
    check_is_fitted(model, 'vocabulary_')
    vocabulary = model.vocabulary_
    if (not vocabulary or any(type(k) is not str for k in vocabulary)
            or sorted(vocabulary.values()) != list(range(len(vocabulary)))):
        raise ValueError('Contiguous unique fitted vocabulary required')
    words = [k for k, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    params = _parameters(model)
    idf = None
    if type(model) is TfidfVectorizer and model.use_idf:
        values = np.asarray(model.idf_)
        if (values.dtype != np.dtype(params['dtype']) or values.shape != (len(words),)
                or not np.isfinite(values).all() or (values <= 0).any()):
            raise ValueError('Finite positive aligned IDF required')
        idf = values.tolist()
    return dict(format='sciona.text_vectorizer_prediction.v1',
                mode='tfidf' if type(model) is TfidfVectorizer else 'count',
                sklearn_version=version('scikit-learn'), parameters=params,
                vocabulary=words, idf=idf)


def restore_text_vectorizer(state):
    fields = {'format', 'mode', 'sklearn_version', 'parameters', 'vocabulary', 'idf'}
    if (type(state) is not dict or set(state) != fields
            or state['format'] != 'sciona.text_vectorizer_prediction.v1'
            or state['mode'] not in ('count', 'tfidf')
            or state['sklearn_version'] != version('scikit-learn')):
        raise ValueError('Known exact-version text encoder state required')
    # Reject non-JSON objects and nonfinite numeric controls before reconstruction.
    try:
        params = json.loads(json.dumps(state['parameters'], allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError('JSON-only text encoder parameters required') from error
    cls = CountVectorizer if state['mode'] == 'count' else TfidfVectorizer
    if type(params) is not dict or set(params) != set(cls().get_params(deep=False)):
        raise ValueError('Exact encoder parameter fields required')
    if params['vocabulary'] is not None or params['dtype'] not in ('float32', 'float64'):
        raise ValueError('Known dtype and canonical vocabulary representation required')
    words = state['vocabulary']
    if (type(words) is not list or not words or any(type(word) is not str for word in words)
            or len(set(words)) != len(words)):
        raise ValueError('Nonempty ordered unique vocabulary required')
    tokenizer = params['tokenizer']
    if tokenizer is not None:
        if (type(tokenizer) is not dict or set(tokenizer) != {'kind', 'word_characters', 'whitespace_characters'}
                or tokenizer['kind'] != 'character_runs'):
            raise ValueError('Known tokenizer policy required')
        params['tokenizer'] = CharacterRunTokenizer(tokenizer['word_characters'], tokenizer['whitespace_characters'])
    params['dtype'] = np.dtype(params['dtype']).type
    if type(params['ngram_range']) is not list or len(params['ngram_range']) != 2:
        raise ValueError('Explicit ngram range required')
    params['ngram_range'] = tuple(params['ngram_range'])
    model = cls(**params)
    model._validate_params()
    # Recheck supported semantic policies after typed reconstruction.
    if _parameters(model) != state['parameters']:
        raise ValueError('Parameter normalization changed state')
    model.vocabulary = {word: i for i, word in enumerate(words)}
    model._validate_vocabulary()
    if state['mode'] == 'tfidf' and model.use_idf:
        values = state['idf']
        if (type(values) is not list or len(values) != len(words)
                or any(type(v) not in (int, float) for v in values)):
            raise ValueError('Aligned numeric IDF required')
        values = np.asarray(values, dtype=model.dtype)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Finite positive IDF required')
        model.idf_ = values
    elif state['idf'] is not None:
        raise ValueError('Unexpected IDF state')
    elif state['mode'] == 'tfidf':
        from sklearn.feature_extraction.text import TfidfTransformer
        model._tfidf = TfidfTransformer(norm=model.norm, use_idf=False,
            smooth_idf=model.smooth_idf, sublinear_tf=model.sublinear_tf)
    if state['mode'] == 'tfidf':
        model._tfidf.n_features_in_ = len(words)
    return model

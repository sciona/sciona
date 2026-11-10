"""Explicit serial adapter for the pinned Mercari PJ application policy.

Adapted from pjankiewicz/mercari-solution commit
374301ad1c32cbc93dcc40313d5d7bb9c5503746, mercari/transformers.py.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

This policy is intentionally lossy and application-specific. English stemming,
ASCII transliteration, category conventions and source field bindings must
not be inferred for unrelated domains. Dependency publication review remains
separate. No processes are started, including when stemming is enabled.
"""
from importlib.metadata import version

from sklearn.base import BaseEstimator, TransformerMixin
from sciona.mercari_table_preprocessing import _input
from sciona.mercari_text_cleaning import clean_source_text


class MercariPJPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, n_jobs=1, hashchars=False, stem=True):
        self.n_jobs = n_jobs
        self.hashchars = hashchars
        self.stem = stem

    def _configuration(self):
        if type(self.n_jobs) is not int or self.n_jobs != 1:
            raise ValueError('Serial PJ policy requires n_jobs=1')
        if type(self.hashchars) is not bool or type(self.stem) is not bool:
            raise ValueError('Explicit boolean cleaning modes required')

    def fit(self, frame, y=None):
        self._configuration()
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, frame):
        self._configuration()
        result = _input(frame)
        result['item_condition_id'] = result['item_condition_id'].fillna('UNK').astype(str)
        result['shipping'] = result['shipping'].astype(str)
        result.loc[result['item_description'] == 'No description yet', 'item_description'] = 'unk'
        result['item_description'] = result['item_description'].fillna('').astype(str)
        result['name'] = result['name'].fillna('').astype(str)
        for name, limit in (('item_description', 1050), ('name', 100), ('brand_name', 100)):
            result[name] = result[name].map(lambda value: value[:limit] if isinstance(value, str) else value)
        if self.stem:
            for source, destination in (('name', 'name_clean'), ('item_description', 'desc_clean'),
                                        ('brand_name', 'brand_name_clean'), ('category_name', 'category_name_clean')):
                result[destination] = [clean_source_text(value, hashchars=self.hashchars) for value in result[source]]
        result['no_cat'] = result['category_name'].isnull().map(int)
        fallback = ['unk', 'unk', 'unk']
        result['cat_split'] = result['category_name'].fillna('/'.join(fallback)).map(lambda value: value.split('/'))
        for index in range(3):
            # Preserve source list fallback: pandas string lowering maps it to NaN.
            result[f'cat_{index + 1}'] = result['cat_split'].map(
                lambda value: value[index] if isinstance(value, list) and len(value) > index else fallback).str.lower()
        result['is_bundle'] = (result['item_description'].str.find('bundl') >= 0).map(int)
        return result


def _runtime():
    return {name: version(name) for name in ('pandas', 'nltk', 'Unidecode')}


def export_pj_preprocessor(model):
    if type(model) is not MercariPJPreprocessor:
        raise ValueError('Known PJ application policy required')
    model._configuration()
    return dict(format='sciona.mercari_pj_policy.v1', runtime=_runtime(),
                n_jobs=1, hashchars=model.hashchars, stem=model.stem)


def restore_pj_preprocessor(state):
    if (type(state) is not dict or set(state) != {'format', 'runtime', 'n_jobs', 'hashchars', 'stem'}
            or state['format'] != 'sciona.mercari_pj_policy.v1' or state['runtime'] != _runtime()):
        raise ValueError('Known exact-runtime PJ policy required')
    result = MercariPJPreprocessor(state['n_jobs'], state['hashchars'], state['stem'])
    result._configuration()
    return result

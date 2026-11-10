"""Explicit adapter for the pinned Mercari KL table policy, not generic cleaning.

Adapted from pjankiewicz/mercari-solution commit
374301ad1c32cbc93dcc40313d5d7bb9c5503746, mercari/transformers.py.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

Public source interface names and category rules are intentional application
policy. Other applications must explicitly map their inputs and assess these
lossy rules. Reusable feature selection/encoding/composition is separate.
Training labels for the frequency whitelist must be strings or missing;
non-string categorical labels are outside this adapter's qualified contract.
"""
import json
import re
from importlib.metadata import version

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sciona.table_feature_operations import _frame

_FIELDS = ('name', 'item_description', 'category_name', 'brand_name', 'shipping', 'item_condition_id')


def _input(frame):
    _frame(frame)
    if any(name not in frame for name in _FIELDS):
        raise ValueError('Pinned application interface fields required')
    return frame.copy(deep=True)


class MercariFillEmpty(TransformerMixin, BaseEstimator):
    def fit(self, frame, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, frame):
        result = _input(frame)
        for name in _FIELDS:
            result[name] = result[name].fillna(0 if name == 'shipping' else 'unk')
        return result


class MercariKLPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, num_brands, repl_patterns):
        self.num_brands = num_brands
        self.repl_patterns = repl_patterns
        self._popular = None

    def _configuration(self):
        if type(self.num_brands) is not int or self.num_brands < 1:
            raise ValueError('Positive top-category count required')
        if type(self.repl_patterns) is not list:
            raise ValueError('Explicit ordered replacement list required')
        for pair in self.repl_patterns:
            if type(pair) not in (list, tuple) or len(pair) != 2 or any(type(v) is not str for v in pair):
                raise ValueError('String regex/replacement pairs required')
            re.compile(pair[0], re.IGNORECASE)

    def fit(self, frame, y=None):
        self._popular = None
        self._configuration()
        values = _input(frame)['brand_name']
        if not all(isinstance(value, str) for value in values.dropna()):
            raise ValueError('String or missing category labels required')
        self._popular = values.value_counts().index[:self.num_brands].tolist()
        return self

    def transform(self, frame):
        self._configuration()
        if self._popular is None:
            raise ValueError('Training-only category whitelist required')
        result = _input(frame)
        for name in ('category_name', 'brand_name', 'item_description', 'name'):
            result[name] = result[name].fillna('unknown' if name in ('category_name', 'brand_name') else '').map(str)
        for name, limit in (('item_description', 1050), ('name', 100), ('brand_name', 100)):
            result[name] = result[name].map(lambda value: value[:limit])
        # Source fits the whitelist before trimming, then compares trimmed values.
        result.loc[~result['brand_name'].isin(self._popular), 'brand_name'] = 'Other'
        parts = result['category_name'].str.split('/')
        result['category_name_l1'] = parts.map(lambda value: value[0])
        result['category_name_l1s'] = parts.map(lambda value: value[0] if value[0] != 'Women' else '/'.join(value[:2]))
        result['category_name_l2'] = parts.map(lambda value: '/'.join(value[:2]))
        for pattern, replacement in self.repl_patterns:
            result['item_description'] = result['item_description'].str.replace(pattern, replacement, flags=re.IGNORECASE, regex=True)
        missing = result['item_description'] == 'No description yet'
        result.loc[missing, 'item_description'] = ''
        result['no_description'] = missing.astype(str)
        result['item_condition_id'] = result['item_condition_id'].map(str)
        result['shipping'] = result['shipping'].map(str)
        return result


def export_kl_preprocessor(model):
    if type(model) is not MercariKLPreprocessor or model._popular is None:
        raise ValueError('Known fitted KL policy required')
    model._configuration()
    return dict(format='sciona.mercari_kl_prediction.v1', pandas_version=version('pandas'),
        num_brands=model.num_brands, repl_patterns=json.loads(json.dumps(model.repl_patterns)),
        popular=list(model._popular))


def restore_kl_preprocessor(state):
    if (type(state) is not dict or set(state) != {'format', 'pandas_version', 'num_brands', 'repl_patterns', 'popular'}
            or state['format'] != 'sciona.mercari_kl_prediction.v1' or state['pandas_version'] != version('pandas')):
        raise ValueError('Known exact-runtime KL state required')
    model = MercariKLPreprocessor(state['num_brands'], json.loads(json.dumps(state['repl_patterns'])))
    model._configuration()
    popular = state['popular']
    if (type(popular) is not list or any(type(v) is not str for v in popular)
            or len(set(popular)) != len(popular) or len(popular) > model.num_brands):
        raise ValueError('Unique string whitelist within configured limit required')
    model._popular = list(popular)
    return model

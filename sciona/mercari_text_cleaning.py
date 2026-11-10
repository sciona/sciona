"""Serial adapter preserving the pinned Mercari text-cleaning recipe.

Adapted from mercari/text_utils.py in pjankiewicz/mercari-solution,
commit 374301ad1c32cbc93dcc40313d5d7bb9c5503746.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

This policy is lossy: ASCII transliteration, English Porter stemming, emoticon
replacement and optional character-set collapsing are not language-neutral.
Callers must choose this adapter explicitly. Dependency qualification is for
the pinned contemporary runtime, not the unrecovered historical environment.
"""
from functools import lru_cache
import re

from nltk.stem import PorterStemmer
from unidecode import unidecode

from sciona.mercari_text_policy import source_tokenizer

_SPACES = re.compile(r'\s\s+')
_DASHES = re.compile('[-]{2,}')
_TOKENIZER = source_tokenizer()
_STEMMER = PorterStemmer()


@lru_cache(maxsize=65536)
def _stem(token):
    return _STEMMER.stem(token)


def clean_source_text(text, *, hashchars=False):
    """Clean a string serially; missing-value coercion belongs to the table adapter."""
    if not isinstance(text, str):
        raise ValueError('Text input required; resolve missing values upstream')
    if type(hashchars) is not bool:
        raise ValueError('hashchars must be boolean')
    text = _SPACES.sub(' ', text.lower())
    text = unidecode(text)
    text = text.replace(' -', ' ').replace('- ', ' ').replace(' - ', ' ')
    text = _DASHES.sub(' ', text)
    text = text.replace(':)', 'smiley').replace('(:', 'smiley').replace(':-)', 'smiley')
    tokens = _TOKENIZER(text.lower())
    if hashchars:
        tokens = [''.join(sorted(set(token))) for token in tokens]
    return ''.join(_stem(token) for token in tokens).lower()

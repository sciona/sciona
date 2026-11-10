"""Configurable Unicode character-run tokenization without worker processes.

Adapted from FastTokenizer in pjankiewicz/mercari-solution at commit
374301ad1c32cbc93dcc40313d5d7bb9c5503746, mercari/text_utils.py.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

Word and whitespace alphabets are explicit caller policy. Characters outside
both alphabets form individual tokens, including repeated punctuation. No
normalization, stemming, case folding, transliteration or whitespace removal
is implied. This is a code-point tokenizer, not a grapheme-cluster tokenizer.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CharacterRunTokenizer:
    word_characters: str
    whitespace_characters: str
    _words: frozenset = field(init=False, repr=False, compare=False)
    _spaces: frozenset = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.word_characters, str) or not isinstance(self.whitespace_characters, str):
            raise ValueError('Character alphabets must be strings')
        words, spaces = frozenset(self.word_characters), frozenset(self.whitespace_characters)
        if words & spaces:
            raise ValueError('Word and whitespace alphabets must be disjoint')
        object.__setattr__(self, '_words', words)
        object.__setattr__(self, '_spaces', spaces)

    def __call__(self, text):
        if not isinstance(text, str):
            raise ValueError('Tokenizer input must be text')
        tokens, run = [], []
        previous_kind = None
        for char in text:
            kind = 1 if char in self._words else 2 if char in self._spaces else 0
            if run and (kind == 0 or kind != previous_kind):
                tokens.append(''.join(run))
                run = []
            run.append(char)
            previous_kind = kind
        if run:
            tokens.append(''.join(run))
        return tokens

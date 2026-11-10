import pickle

import pytest

from sciona.character_run_tokenizer import CharacterRunTokenizer


def test_runs_preserve_punctuation_and_whitespace():
    tokenize = CharacterRunTokenizer('abcABC012-&', ' \t\n')
    assert tokenize('Ab-1&c!! \t\n2?') == ['Ab-1&c', '!', '!', ' \t\n', '2', '?']
    assert tokenize('') == []
    assert tokenize('!!!') == ['!', '!', '!']


def test_unicode_policy_is_explicit_and_lossless():
    text = 'αβ α\u0301🙂🙂\r\n'
    tokenizer = CharacterRunTokenizer('αβ', ' \n')
    expected = ['αβ', ' ', 'α', '\u0301', '🙂', '🙂', '\r', '\n']
    assert tokenizer(text) == expected
    assert ''.join(tokenizer(text)) == text
    assert pickle.loads(pickle.dumps(tokenizer))(text) == expected


def test_cross_domain_alphabet_adapters():
    sequence = CharacterRunTokenizer('ACGT', ' ')
    expression = CharacterRunTokenizer('xyz0123456789', ' ')
    assert sequence('ACG?TT') == ['ACG', '?', 'TT']
    assert expression('x12+y3') == ['x12', '+', 'y3']
    assert sequence('x12+y3') == list('x12+y3')


def test_long_run():
    assert CharacterRunTokenizer('a', ' ')('a' * 100000) == ['a' * 100000]


@pytest.mark.parametrize('words,spaces', [('a', 'a'), (['a'], ' '), ('a', None)])
def test_invalid_alphabets(words, spaces):
    with pytest.raises(ValueError):
        CharacterRunTokenizer(words, spaces)


def test_nontext_rejected():
    with pytest.raises(ValueError):
        CharacterRunTokenizer('a', ' ')(None)

import pytest

from sciona.mercari_text_cleaning import clean_source_text


def test_explicit_lossy_language_policy():
    assert clean_source_text('Café running :)') == 'cafe run smiley'
    assert clean_source_text('') == ''
    assert clean_source_text('A!!') == 'a!!'


def test_character_set_option_is_separate():
    assert clean_source_text('bbaa', hashchars=True) == 'ab'
    assert clean_source_text('bbaa') == 'bbaa'


def test_repeatability_and_no_cross_call_policy_leakage():
    assert clean_source_text('bbaa', hashchars=True) == 'ab'
    assert clean_source_text('bbaa', hashchars=False) == 'bbaa'
    assert clean_source_text('Café running :)') == clean_source_text('Café running :)')


@pytest.mark.parametrize('value', [None, 1, ['text'], b'text'])
def test_table_coercion_is_not_implicit(value):
    with pytest.raises(ValueError):
        clean_source_text(value)


@pytest.mark.parametrize('value', [1, None, 'false'])
def test_invalid_mode_rejected(value):
    with pytest.raises(ValueError):
        clean_source_text('text', hashchars=value)

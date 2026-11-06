"""Synthetic spelling, domain and synonym policy with caller-state isolation."""
import pytest
from sciona.text_pair_normalization_policy import NormalizationPolicy
from sciona.atoms.ml.text_nlp.atoms import clean_text


def policy():
    return NormalizationPolicy({'colour':10,'cat':8,'pet':7,'animal':6},
                               {'colour':['color']},{'cat':['pet','cat']})


def test_complete_normalization_and_existing_cleaner():
    calls=[]
    def checked(text,operations,vocabulary):
        calls.append(operations)
        return clean_text(text,operations,vocabulary)
    assert policy().normalize_pairs([['ＣＯＬＯＵＲ!!!','cta']],cleaner=checked)==[('color','cat pet')]
    assert calls==[['spell'],['spell']]


def test_policy_copies_caller_state_and_preserves_pair_order():
    vocab={'cat':8,'pet':7,'animal':6};domain={'cat':['pet']};synonyms={'pet':['animal']}
    p=NormalizationPolicy(vocab,domain,synonyms)
    vocab.clear();domain['cat'].append('cat');synonyms['pet'].clear()
    assert p.normalize_pairs([['cat','animal']])==[('pet animal','animal')]
    assert p.normalize_pairs([['animal','cat']])==[('animal','pet animal')]
    with pytest.raises(TypeError):p.vocabulary['cat']=0


def test_expansion_is_one_pass_and_population_independent():
    p=NormalizationPolicy({'cat':8,'pet':7,'animal':6},{},{'cat':['pet'],'pet':['animal']})
    expected=p.normalize_pairs([['cat','pet']])
    assert expected==[('cat pet','pet animal')]
    assert p.normalize_pairs([['animal','animal'],['cat','pet']])[1:]==expected


@pytest.mark.parametrize('vocabulary,domain,synonyms',[
    ({},{},{}),({'Cat':1},{},{}),({'cat':True},{},{}),({'cat':1},{'cat':[]},{}),
    ({'cat':1},{},{'cat':['two words']}),({'cat':1},{},{'cat':['cat']*513})])
def test_invalid_policy(vocabulary,domain,synonyms):
    with pytest.raises(ValueError):NormalizationPolicy(vocabulary,domain,synonyms)


@pytest.mark.parametrize('pairs',[[],[['cat']],[['','cat']],[['!!!','cat']],[['cat',1]],[['x'*513,'cat']]])
def test_invalid_pairs(pairs):
    with pytest.raises(ValueError):policy().normalize_pairs(pairs)


def test_expansion_limit_rejects_without_truncation():
    p=NormalizationPolicy({'cat':1},{},{'cat':['animal']})
    with pytest.raises(ValueError):p.normalize_pairs([[' '.join(['cat']*100),'cat']])


def test_split_collision_detected_after_spelling_and_domain_mapping():
    p=policy()
    with pytest.raises(ValueError,match='Overlapping'):
        p.normalize_populations([['colour','cat']],[['cta','ＣＯＬＯＵＲ']],[['animal','animal']])


def test_fitted_duplicates_rejected_after_mapping():
    with pytest.raises(ValueError,match='Duplicate'):
        policy().normalize_populations([['colour','cat'],['cta','colour']],
                                      [['pet','animal']],[['animal','animal']])


def test_disjoint_populations_keep_row_order():
    p=policy()
    inputs=([['colour','cat']],[['pet','animal']],[['animal','animal'],['cat','cat']])
    assert p.normalize_populations(*inputs)==tuple(p.normalize_pairs(x) for x in inputs)

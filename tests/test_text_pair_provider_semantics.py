"""Synthetic cross-domain numerical and text semantics beyond recipe parity."""
from functools import lru_cache
from itertools import product
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sciona.atoms.ml.text_nlp.atoms import clean_text, levenshtein
from sciona.atoms.ml.sklearn.isotonic.atoms import isotonic_regression_fit, isotonic_regression_transform


def test_edit_distance_exhaustive_small_unicode_strings():
    @lru_cache(None)
    def reference(a, b):
        if not a: return len(b)
        if not b: return len(a)
        return min(reference(a[1:], b) + 1, reference(a, b[1:]) + 1,
                   reference(a[1:], b[1:]) + (a[0] != b[0]))
    strings = [''] + [''.join(x) for n in (1, 2, 3) for x in product('aé', repeat=n)]
    for a in strings:
        for b in strings:
            assert levenshtein(a, b) == reference(a, b)


def test_cleaning_order_and_nonprintable_handling_are_explicit():
    text = '<b>ＣＡＴ</b>\x00\nhttps://example.invalid/x  DOG'
    operations = ['lower', 'url', 'unicode', 'html']
    assert clean_text(text, operations) == 'cat dog'
    assert operations == ['lower', 'url', 'unicode', 'html']


def test_spelling_ties_and_caller_vocabulary_preserved():
    vocab = {'cat': 2, 'bat': 2}
    assert clean_text('dat', ['spell'], vocab) == 'cat'
    assert vocab == {'cat': 2, 'bat': 2}


@pytest.mark.parametrize('x,y,weights,increasing', [
    ([0.,1.,1.,3.], [4.,2.,6.,8.], [1.,2.,1.,3.], True),
    ([0.,1.,2.,3.], [8.,4.,6.,2.], [1.,1.,1.,1.], False),
    ([0.,1.,2.,3.], [8.,4.,6.,2.], [1.,0.,1.,1.], False),
])
def test_nonprobability_weighted_isotonic_data(x,y,weights,increasing):
    x,y,w = (np.array(v,dtype=np.float64) for v in (x,y,weights))
    snapshots = [v.copy() for v in (x,y,w)]
    state = isotonic_regression_fit(x,y,sample_weight=w,increasing=increasing,
                                    y_min=1.,y_max=7.,out_of_bounds='clip')
    reference = IsotonicRegression(increasing=increasing,y_min=1.,y_max=7.,out_of_bounds='clip').fit(x,y,sample_weight=w)
    query = np.array([-1.,.5,1.,2.5,4.])
    np.testing.assert_allclose(isotonic_regression_transform(query,state),reference.predict(query),rtol=1e-14,atol=1e-14)
    for actual,before in zip((x,y,w),snapshots,strict=True):
        np.testing.assert_array_equal(actual,before)


@pytest.mark.parametrize('x,y', [([0.,np.nan],[0.,1.]), ([0.,1.],[0.,np.inf])])
def test_nonfinite_isotonic_fit_rejects(x,y):
    with pytest.raises(ValueError):
        isotonic_regression_fit(np.array(x),np.array(y))

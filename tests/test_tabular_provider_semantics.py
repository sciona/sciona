"""Cross-domain numerical checks using exclusively synthetic inputs."""
import numpy as np
import pytest
from sciona.atoms.ml.tabular.gradient_boosting.atoms import (
    frequency_encode_fit, frequency_encode, target_encode, group_aggregate,
)
from sciona.atoms.ml.sklearn.decomposition.lda_shell.atoms import lda_normalize_document_topics
from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping.atoms import rfe_active_feature_indices


def test_frequency_counts_and_unknown_lookup_are_separate_operations():
    fitted = frequency_encode_fit(np.array(['b', 'a', 'b', 'b'], dtype=object))
    assert fitted == {'a': .25, 'b': .75}
    np.testing.assert_array_equal(frequency_encode(np.array(['b', 'new', 'a', 'b'], dtype=object), fitted, 2.),
        [.75, 2., .25, .75])
    assert fitted == {'a': .25, 'b': .75}


def test_target_statistics_accept_real_values_and_explicit_prior():
    keys = np.array(['b', 'a', 'b'], dtype=object); targets = np.array([-10., 40., 20.])
    fitted = target_encode(keys, targets, smoothing=2., prior=-4.)
    assert fitted == {'a': 32/3, 'b': .5}
    assert target_encode(keys, targets, smoothing=0., prior=99.) == {'a': 40., 'b': 5.}
    inferred = target_encode(keys, targets, smoothing=1., prior=None)
    assert inferred == {'a': (40+50/3)/2, 'b': (10+50/3)/3}


@pytest.mark.parametrize('stat, expected', [
    ('count', [1, 3]), ('sum', [-4, 14]), ('mean', [-4, 14/3]),
    ('var', [0, 128/9]), ('std', [0, np.sqrt(128/9)]), ('min', [-4, 2]),
    ('max', [-4, 10]), ('first', [-4, 2]), ('last', [-4, 10]),
])
def test_group_statistics_sorted_key_order_and_population_denominator(stat, expected):
    # Input order is deliberately different from sorted unique-key order.
    result = group_aggregate(np.array([2., -4., 2., 10.]), np.array(['b', 'a', 'b', 'b'], dtype=object), stat)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)


def test_row_normalization_is_generic_nonnegative_mass_normalization():
    x = np.array([[2., 0., 6.], [1., 2., 1.]])
    before = x.copy()
    np.testing.assert_array_equal(lda_normalize_document_topics(x), [[.25, 0., .75], [.25, .5, .25]])
    np.testing.assert_array_equal(lda_normalize_document_topics(x*np.array([[3.], [9.]])),
        lda_normalize_document_topics(x))
    np.testing.assert_array_equal(x, before)


def test_active_indices_preserve_original_order():
    mask = np.array([False, True, False, True, True])
    result = rfe_active_feature_indices(mask)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, [1, 3, 4])
    np.testing.assert_array_equal(mask, [False, True, False, True, True])

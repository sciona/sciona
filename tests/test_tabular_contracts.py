import numpy as np
import pandas as pd
import pytest

from sciona.tabular_contracts import stable_feature_order, ordered_feature_matrix


def test_order_preserves_first_occurrence_and_does_not_mutate():
    names = ['z', 'a', 'z', 'b']
    assert stable_feature_order(names) == ['z', 'a', 'b']
    assert names == ['z', 'a', 'z', 'b']


def test_matrix_uses_persisted_order_and_owns_output():
    frame = pd.DataFrame({'b': [2.], 'ignored': [9.], 'a': [1.]})
    result = ordered_feature_matrix(frame, ('a', 'b'))
    np.testing.assert_array_equal(result, [[1., 2.]])
    np.testing.assert_array_equal(result, ordered_feature_matrix(frame[frame.columns[::-1]], ('a', 'b')))
    result[0, 0] = 99
    assert frame.a.iloc[0] == 1


@pytest.mark.parametrize('frame,names', [
    (pd.DataFrame({'a': [1.]}), ['a', 'missing']),
    (pd.DataFrame({'a': [1.]}), ['a', 'a']),
    (pd.DataFrame([[1., 2.]], columns=['a', 'a']), ['a']),
    (pd.DataFrame({'a': [float('nan')]}), ['a']),
    (pd.DataFrame({'a': [float('inf')]}), ['a']),
    (pd.DataFrame({'a': ['1']}), ['a']),
    (pd.DataFrame({'a': [1j]}), ['a']),
    (pd.DataFrame({'a': [1.]}), []),
])
def test_invalid_matrix_contract_rejected(frame, names):
    with pytest.raises(ValueError):
        ordered_feature_matrix(frame, names)

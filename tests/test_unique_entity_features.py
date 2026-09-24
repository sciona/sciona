import numpy as np
import pandas as pd
import pytest

from sciona.unique_entity_features import unique_entity_features


def test_machine_snapshot_reuse_and_query_order():
    records = np.array(['unit_a', 'unit_b'])
    attributes = pd.DataFrame({'mode': ['active', 'unknown'], 'capacity': [10, 20]}, index=[8, 3])
    before = attributes.copy(deep=True)
    result = unique_entity_features(np.array(['unit_b', 'missing', 'unit_a', 'unit_a']),
        records, attributes, {'mode': ['active', 'idle']}, 'OTHER')
    assert result['mode'].tolist() == ['OTHER', 'OTHER', 'active', 'active']
    np.testing.assert_allclose(result['capacity'], [20, np.nan, 10, 10], equal_nan=True)
    pd.testing.assert_frame_equal(attributes, before)
    assert result.index.tolist() == [0, 1, 2, 3]


def test_duplicate_records_rejected_even_when_identical():
    with pytest.raises(ValueError, match='Unique authoritative'):
        unique_entity_features(np.array(['a']), np.array(['a', 'a']),
            pd.DataFrame({'mode': ['on', 'on']}), {'mode': ['on']}, 'OTHER')


def test_empty_snapshot_and_queries():
    attributes = pd.DataFrame({'mode': pd.Series(dtype=str), 'value': pd.Series(dtype=float)})
    result = unique_entity_features(np.array(['a']), np.array([], dtype=str), attributes, {'mode': []}, 'missing')
    assert result['mode'].tolist() == ['missing']
    assert result['value'].isna().all()
    empty = unique_entity_features(np.array([], dtype=str), np.array([], dtype=str), attributes, {'mode': []}, 'missing')
    assert empty.shape == (0, 2)


def test_categorical_storage_accepts_fallback_not_in_original_categories():
    result = unique_entity_features(np.array(['a']), np.array(['a']),
        pd.DataFrame({'mode': pd.Categorical(['on'])}), {'mode': ['off']}, 'missing')
    assert result['mode'].tolist() == ['missing']


@pytest.mark.parametrize('vocab', [{'missing': []}, {'mode': ['on', 'on']}, {'mode': 'on'}])
def test_invalid_vocabulary(vocab):
    with pytest.raises(ValueError):
        unique_entity_features(np.array(['a']), np.array(['a']), pd.DataFrame({'mode': ['on']}), vocab, 'OTHER')

import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sciona.table_feature_operations import (
    TableColumnSelector, TableTextConcat, TableToRecords, NumericCSR,
    export_table_operation, restore_table_operation,
)


def test_explicit_column_order_vector_and_no_mutation():
    frame = pd.DataFrame({'b': [2, 4], 'a': [1, 3]}, index=[7, 2])
    declared = TableColumnSelector(['a', 'b']).transform(frame)
    assert list(declared.columns) == ['a', 'b'] and list(declared.index) == [7, 2]
    assert list(TableColumnSelector(['a', 'b'], order='frame').transform(frame).columns) == ['b', 'a']
    assert TableColumnSelector(['a']).transform(frame) == [1, 3]
    declared.iloc[0, 0] = 99
    assert frame.iloc[0]['a'] == 1


def test_text_prefix_policy_and_empty_inputs():
    frame = pd.DataFrame({'first': ['α', ''], 'second': ['β', '!']}, index=[3, 1])
    operation = TableTextConcat(['second', 'first'], ['[right]', '[left]'], 'joined')
    result = operation.transform(frame)
    assert list(result['joined']) == ['[right]β[left]α', '[right]![left]']
    assert 'joined' not in frame and list(result.index) == [3, 1]
    assert operation.transform(frame.iloc[:0]).shape == (0, 3)
    assert TableToRecords().transform(frame) == [{'first': 'α', 'second': 'β'}, {'first': '', 'second': '!'}]


@pytest.mark.parametrize('operation,values', [
    (TableColumnSelector(['a']), pd.DataFrame({'a': ['x', 'y']})),
    (TableTextConcat(['a'], ['prefix'], 'b'), pd.DataFrame({'a': ['x', 'y']})),
    (TableToRecords(), pd.DataFrame({'a': ['x', 'y']})),
    (NumericCSR('float64'), np.array([[0., 1.], [2., 0.]])),
])
def test_json_configuration_roundtrip(operation, values):
    restored = restore_table_operation(json.loads(json.dumps(export_table_operation(operation))))
    actual, expected = restored.transform(values), operation.transform(values)
    if isinstance(actual, pd.DataFrame): pd.testing.assert_frame_equal(actual, expected)
    elif sparse.issparse(actual):
        for key in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    else: assert actual == expected


@pytest.mark.parametrize('operation,values', [
    (TableColumnSelector(['missing']), pd.DataFrame({'a': [1]})),
    (TableColumnSelector(['a', 'a']), pd.DataFrame({'a': [1]})),
    (TableToRecords(), pd.DataFrame({'a': [[1]]})),
    (TableTextConcat(['a'], [''], 'b'), pd.DataFrame({'a': [None]})),
    (TableTextConcat(['a'], [''], 'a'), pd.DataFrame({'a': ['x']})),
    (NumericCSR(), np.array([[1j]])),
])
def test_invalid_contract_rejected(operation, values):
    with pytest.raises(ValueError): operation.transform(values)


def test_unknown_state_and_parameters_rejected():
    state = export_table_operation(TableToRecords())
    state['kind'] = 'import_module'
    with pytest.raises(ValueError): restore_table_operation(state)
    state = export_table_operation(TableToRecords())
    state['parameters']['extra'] = True
    with pytest.raises(ValueError): restore_table_operation(state)

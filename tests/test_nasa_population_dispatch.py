import numpy as np
import pandas as pd
import pytest

from sciona.nasa_population_dispatch import partition_records, assemble_predictions


def records():
    queries = pd.DataFrame(dict(gufi=['AAA01', 'BBB02', 'AAA03', 'BBB04'],
        timestamp=pd.to_datetime(['2020-01-01'] * 4), airport=['KAAA', 'KBBB', 'KAAA', 'KBBB']))
    # History entities deliberately differ from the requested entities.
    return dict(queries=queries, estimates=pd.DataFrame({'gufi':['EST99']}),
                stands=pd.DataFrame({'gufi':['HIS99']}), arrivals=pd.DataFrame({'gufi':['HIS99']}))


def predictions(partitions):
    result = {}
    for airport, tables in partitions.items():
        frame = tables['queries'].copy()
        frame['minutes_until_pushback'] = frame.gufi.str[-2:].astype(np.int32)
        result[airport] = frame.iloc[::-1].reset_index(drop=True)
    return result


def test_interleaved_queries_restore_order_without_losing_history():
    source = records()
    parts, ids = partition_records(source, ('KBBB', 'KAAA', 'KZZZ'))
    assert list(parts) == ['KBBB', 'KAAA']
    assert len(parts['KAAA']['stands']) == 1
    assert parts['KAAA']['stands'].gufi.tolist() == ['HIS99']
    output = assemble_predictions(ids, predictions(parts))
    pd.testing.assert_frame_equal(output.iloc[:, :3], source['queries'])
    np.testing.assert_array_equal(output.minutes_until_pushback, np.arange(1, 5, dtype=np.int32))
    parts['KAAA']['queries'].loc[0, 'gufi'] = 'CHANGED'
    assert source['queries'].gufi.iloc[0] == 'AAA01'


@pytest.mark.parametrize('fault', ['missing_slot', 'extra_slot', 'missing_row', 'duplicate_row', 'wrong_identity', 'wrong_slot', 'wrong_dtype'])
def test_reassembly_rejects_incomplete_or_misaligned_predictions(fault):
    parts, ids = partition_records(records(), ('KAAA', 'KBBB'))
    result = predictions(parts)
    if fault == 'missing_slot':
        del result['KAAA']
    elif fault == 'extra_slot':
        result['KZZZ'] = result['KAAA'].iloc[:0]
    elif fault == 'missing_row':
        result['KAAA'] = result['KAAA'].iloc[:1]
    elif fault == 'duplicate_row':
        result['KAAA'] = pd.concat([result['KAAA'].iloc[:1]] * 2)
    elif fault == 'wrong_identity':
        result['KAAA'].loc[0, 'gufi'] = 'OTHER'
    elif fault == 'wrong_slot':
        result['KAAA'].loc[0, 'airport'] = 'KBBB'
    else:
        result['KAAA']['minutes_until_pushback'] = result['KAAA'].minutes_until_pushback.astype(np.float64)
    with pytest.raises(ValueError):
        assemble_predictions(ids, result)


@pytest.mark.parametrize('fault', ['unknown_slot', 'duplicate_query', 'entity_two_airports', 'duplicate_slots'])
def test_partition_rejects_ambiguous_populations(fault):
    source = records()
    airports = ('KAAA', 'KBBB')
    if fault == 'unknown_slot':
        airports = ('KAAA',)
    elif fault == 'duplicate_query':
        source['queries'] = pd.concat([source['queries'], source['queries'].iloc[:1]])
    elif fault == 'entity_two_airports':
        source['queries'].loc[1, 'gufi'] = 'AAA01'
    else:
        airports = ('KAAA', 'KAAA')
    with pytest.raises(ValueError):
        partition_records(source, airports)

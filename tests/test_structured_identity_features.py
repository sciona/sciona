import numpy as np
import pandas as pd
import pytest

from sciona.structured_identity_features import structured_identity_features as compute


def args():
    known='M123|line|20000101|1200|tail'
    unknown='U999|other|20000101|1200|tail'
    return [np.array([known,known,unknown]),pd.DatetimeIndex(['2000-01-01 11:00','2000-01-01 13:00','2000-01-01 12:00']),np.array([known,unknown])]


def policy():
    return dict(delimiter='|',minimum_parts=5,primary_part=0,category_part=1,
        reference_parts=[2,3],reference_format='%Y%m%d%H%M',
        vocabularies={'primary':['M123'],'numeric':['123'],'category':['line']},fallback='OTHER',seconds_per_unit=60)


def test_manufacturing_identifier_features_and_fallback_order():
    result=compute(*args(),**policy())
    np.testing.assert_array_equal(result['elapsed'],[-60,60,0])
    assert result['primary'].tolist()==['M123','M123','OTHER']
    assert result['prefix'].tolist()==['M','M','U']
    assert result['numeric'].tolist()==['123','123','OTHER']
    assert result['numeric_last'].tolist()==['3','3','R']
    assert result['numeric_length'].tolist()==[3,3,5]


def test_missing_authoritative_record_keeps_time_missing():
    values=args();values[2]=values[2][:1]
    result=compute(*values,**policy())
    assert pd.isna(result['elapsed'].iloc[2])
    assert pd.isna(result['reference_hour'].iloc[2])


def test_explicit_time_units_and_query_duplicates():
    values=args();options=policy();options['seconds_per_unit']=3600
    result=compute(*values,**options)
    np.testing.assert_array_equal(result['elapsed'],[-1,1,0])
    duplicated=compute(values[0][[1,0,1]],values[1][[1,0,1]],values[2],**options)
    pd.testing.assert_frame_equal(duplicated,result.iloc[[1,0,1]].reset_index(drop=True))


@pytest.mark.parametrize('identifier',['invalid','M123||20000101|1200|tail','M123|line|20001301|1200|tail'])
def test_malformed_identifiers_rejected(identifier):
    values=args();values[0]=np.array([identifier]);values[1]=values[1][:1]
    with pytest.raises(ValueError):compute(*values,**policy())


def test_duplicate_records_rejected():
    values=args();values[2]=values[2][[0,0]]
    with pytest.raises(ValueError,match='Unique authoritative'):compute(*values,**policy())


def test_empty_queries():
    values=args();values[0]=values[0][:0];values[1]=values[1][:0]
    assert compute(*values,**policy()).shape==(0,10)

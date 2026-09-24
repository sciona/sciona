import pandas as pd
import pytest
from sciona.atoms.ml.domain_adapters.first_place_raw_training import attach_target,raw_inputs


def test_duplicate_index_target_attachment_and_nonmutation():
    queries=pd.DataFrame({'gufi':['synthetic-a','synthetic-b'],'timestamp':pd.date_range('2030-01-01',periods=2),
                          'minutes_until_pushback':[5.5,8.]},index=[3,3])
    frame=pd.DataFrame({'synthetic_load':[1,2]},index=[3,3])
    before=frame.copy(deep=True)
    result=attach_target(queries,frame,'minutes_until_pushback')
    assert list(result)==['gufi','timestamp','synthetic_load','minutes_until_pushback']
    assert result.index.tolist()==[3,3]
    assert result['minutes_until_pushback'].tolist()==[5.5,8.]
    pd.testing.assert_frame_equal(frame,before)


def test_misaligned_rows_rejected():
    queries=pd.DataFrame({'gufi':['a','b'],'timestamp':pd.date_range('2030-01-01',periods=2),
                          'minutes_until_pushback':[1.,2.]},index=[1,2])
    with pytest.raises(ValueError,match='identities'):
        attach_target(queries,pd.DataFrame({'x':[1,2]},index=[2,1]),'minutes_until_pushback')


def test_metadata_leak_and_incomplete_populations_rejected():
    queries=pd.DataFrame({'gufi':['a'],'timestamp':[pd.Timestamp('2030-01-01')],'minutes_until_pushback':[1.]})
    with pytest.raises(ValueError,match='metadata'):
        attach_target(queries,pd.DataFrame({'minutes_until_pushback':[1.]}),'minutes_until_pushback')
    with pytest.raises(ValueError,match='population'):
        raw_inputs({})

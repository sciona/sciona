import numpy as np
import pandas as pd
import pytest
from sciona.calendar_time_features import calendar_time_features as features


def test_midnight_midday_and_duplicate_queries():
    holidays=pd.DatetimeIndex(['2000-02-29','2000-03-02'])
    queries=pd.DatetimeIndex(['2000-02-29','2000-02-29 12:00','2000-02-29 12:00'])
    result=features(queries,holidays)
    np.testing.assert_array_equal(result['is_holiday'],[True,True,True])
    np.testing.assert_array_equal(result['days_until_holiday'],[0,1,1])
    np.testing.assert_array_equal(result['week_of_month'],[5,5,5])


def test_iso_week_year_boundary():
    result=features(pd.DatetimeIndex(['2021-01-01']),pd.DatetimeIndex(['2021-01-02']))
    assert result['iso_week'].item()==53
    assert result['weekday'].item()==4


@pytest.mark.parametrize('queries,holidays',[
    (pd.DatetimeIndex(['2000-01-02']),pd.DatetimeIndex(['2000-01-01'])),
    (pd.DatetimeIndex(['2000-01-01']),pd.DatetimeIndex(['2000-01-02','2000-01-02'])),
    (pd.DatetimeIndex(['2000-01-01']),pd.DatetimeIndex(['2000-01-02 12:00'])),
    (pd.DatetimeIndex(['2000-01-01'],tz='UTC'),pd.DatetimeIndex(['2000-01-02'])),
])
def test_invalid_coverage_or_clock(queries,holidays):
    with pytest.raises(ValueError):features(queries,holidays)

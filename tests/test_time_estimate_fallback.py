import numpy as np
import pandas as pd
import pytest
from sciona.time_estimate_fallback import grouped_time_estimate_fallback as fallback


def settings():
    return dict(group_column='group',order_column='observed',estimate_column='estimate',query_time_column='queried',seconds_per_unit=60,offset=30,lower=1,upper=299,missing=15)


def test_positional_nonmissing_fractional_and_nonmutation():
    origin=pd.Timestamp('2001-01-01')
    query=pd.DataFrame({'group':['a','unknown','a'],'queried':[origin]*3},index=[9,9,-3])
    history=pd.DataFrame({'group':['a','a'],'observed':[origin,origin+pd.Timedelta(minutes=1)],'estimate':[origin+pd.Timedelta(minutes=75.5),pd.NaT]})
    before=query.copy(deep=True)
    np.testing.assert_array_equal(fallback(query,history,**settings()),[45.5,15,45.5])
    pd.testing.assert_frame_equal(query,before)


@pytest.mark.parametrize('updates',[{'seconds_per_unit':0},{'lower':20},{'missing':float('nan')},{'estimate_column':'absent'}])
def test_invalid_settings(updates):
    origin=pd.Timestamp('2001-01-01')
    query=pd.DataFrame({'group':['a'],'queried':[origin]})
    history=pd.DataFrame({'group':['a'],'observed':[origin],'estimate':[origin]})
    with pytest.raises(ValueError):fallback(query,history,**{**settings(),**updates})

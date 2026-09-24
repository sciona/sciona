import runpy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from sciona.nasa_first_feature_adapter import nasa_first_feature_tables
from sciona.nasa_first_prediction import nasa_first_prediction


def inputs():
    queries,options=runpy.run_path(str(Path(__file__).with_name('test_nasa_first_feature_adapter.py')))['fixture']()
    raw=nasa_first_feature_tables(queries,**options)
    order=sorted(set(raw)-set(queries))
    categorical={c:'OTHER' for c in order if '_cat_' in c}
    numeric={c:0 for c in order if c not in categorical}
    return queries,dict(raw_options=options,feature_columns=order,numeric_fills=numeric,categorical_fills=categorical)


class Predictor:
    def __init__(self,value,columns):self.value=value;self.feature_names_=columns
    def predict(self,features,*,thread_count):
        assert thread_count==1
        return np.full(len(features),self.value)


@pytest.mark.parametrize('population,expected',[('synthetic',65),('kphx',40)])
def test_full_primary_arithmetic_and_override(population,expected):
    queries,options=inputs();before=queries.copy(deep=True)
    models={0:Predictor(10,['etd_time_till_est_dep']),1:Predictor(40,['etd_time_till_est_dep']),
        2:Predictor(10,['etd_time_till_est_dep']),'global_model':Predictor(80,['feat_cat_airport'])}
    output,route=nasa_first_prediction(queries,models,population,**options)
    assert route=='primary'
    np.testing.assert_array_equal(output['minutes_until_pushback'],expected)
    pd.testing.assert_frame_equal(queries,before)


def test_missing_model_uses_available_baseline_with_original_index():
    queries,options=inputs()
    output,route=nasa_first_prediction(queries,{},'synthetic',**options)
    assert route=='baseline'
    np.testing.assert_array_equal(output['minutes_until_pushback'],[30,30,30])
    assert output.index.equals(queries.index)


def test_double_failure_and_empty_queries():
    queries,options=inputs();options['raw_options']={}
    output,route=nasa_first_prediction(queries,{},'synthetic',**options)
    assert route=='constant'
    np.testing.assert_array_equal(output['minutes_until_pushback'],30)
    output,route=nasa_first_prediction(queries.iloc[:0],{},'synthetic',**options)
    assert route=='empty' and output.empty

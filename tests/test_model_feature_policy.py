import numpy as np
import pandas as pd
import pytest
from sciona.model_feature_policy import prepare_model_features


def test_fixed_order_fills_category_normalization_and_nonmutation():
    frame=pd.DataFrame({'metadata':['private_runtime']*3,'value':[np.nan,1.75,-2.75],'category':[1.,np.nan,'known']},index=[8,3,8])
    original=frame.copy(deep=True)
    result=prepare_model_features(frame,['category','value'],{'value':7},{'category':'missing'})
    assert result.columns.tolist()==['category','value']
    assert result['category'].tolist()==['1','missing','known']
    np.testing.assert_array_equal(result['value'],[7,1,-2])
    assert result['value'].dtype==np.int16
    pd.testing.assert_frame_equal(frame,original)
    for i in range(len(frame)):
        pd.testing.assert_frame_equal(result.iloc[[i]],prepare_model_features(frame.iloc[[i]],['category','value'],{'value':7},{'category':'missing'}))


@pytest.mark.parametrize('values',[[40000.],[-40000.],[np.inf],['12']])
def test_invalid_numeric_contract(values):
    with pytest.raises(ValueError):prepare_model_features(pd.DataFrame({'x':values}),['x'],{'x':0},{})


@pytest.mark.parametrize('fills,categories',[({},{}),({'x':0},{'x':'missing'}),({'x':0,'extra':1},{}),({'x':np.nan},{})])
def test_incomplete_conflicting_or_nonfinite_policy(fills,categories):
    with pytest.raises(ValueError):prepare_model_features(pd.DataFrame({'x':[1]}),['x'],fills,categories)


@pytest.mark.parametrize('value',[1.5,np.inf,[],{}])
def test_invalid_category(value):
    with pytest.raises(ValueError):prepare_model_features(pd.DataFrame({'x':[value]}),['x'],{},{'x':'missing'})


def test_nullable_numeric_and_empty_rows():
    frame=pd.DataFrame({'x':pd.Series([1,pd.NA],dtype='Int64')})
    np.testing.assert_array_equal(prepare_model_features(frame,['x'],{'x':3},{})['x'],[1,3])
    assert prepare_model_features(frame.iloc[:0],['x'],{'x':3},{}).shape==(0,1)

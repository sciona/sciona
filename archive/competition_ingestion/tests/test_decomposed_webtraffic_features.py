"""Synthetic exact parity for decomposed source features and generic numeric reuse."""
import numpy as np
import pandas as pd
import pytest
from sciona import decomposed_webtraffic_features as blocks
from sciona.webtraffic_assembly import assemble_features
from sciona.webtraffic_features import batch_autocorr
from scripts.validate_webtraffic_graph import synthetic_payload


@pytest.mark.parametrize('seed',[1,8,23])
@pytest.mark.parametrize('backoffset',[0,17])
def test_complete_feature_parity(seed,backoffset):
    p=synthetic_payload();rng=np.random.RandomState(seed)
    values=rng.poisson(np.arange(4,20)[:,None],size=(16,800)).astype(np.float32)
    values[0]=0;values[1,:3]=np.nan;values[3,10]=np.nan
    frame=pd.DataFrame(values,index=p['pages'],columns=pd.date_range('2001-01-01',periods=800))
    before=frame.copy(deep=True)
    expected,plain=assemble_features(frame,add_days=63,corr_backoffset=backoffset)
    logged,mask,starts,ends=blocks.prepare(frame,0.,63,backoffset)
    corr=blocks.correlations(logged.to_numpy(),starts,ends,[365,91],[1.5,2.],backoffset)
    mapping,encoded=blocks.page_metadata(logged.index.tolist())
    dow,lags,end=blocks.calendar(logged.columns[0],logged.columns[-1],63)
    actual,actual_plain=blocks.assemble(logged,mask,corr,mapping,encoded,blocks.popularity(logged),dow,lags,end)
    assert actual_plain==plain and actual.keys()==expected.keys()
    for key in actual:
        if isinstance(actual[key],pd.DataFrame):pd.testing.assert_frame_equal(actual[key],expected[key],check_exact=True)
        elif isinstance(actual[key],pd.Series):pd.testing.assert_series_equal(actual[key],expected[key],check_exact=True)
        else:np.testing.assert_array_equal(actual[key],expected[key],strict=True)
    pd.testing.assert_frame_equal(frame,before,check_exact=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_generic_lags_and_population(dtype):
    values=np.random.RandomState(91).normal(size=(5,80)).astype(dtype)
    starts=np.zeros(5,dtype=np.int32);ends=np.full(5,79,dtype=np.int32)
    result=blocks.correlations(values,starts,ends,[3,7,19],[1.,2.,1.5],4)
    assert result.shape==(5,3)
    for i,(lag,threshold) in enumerate(zip([3,7,19],[1.,2.,1.5])):
        raw=np.nan_to_num(batch_autocorr(values,lag,starts,ends,threshold,4))
        np.testing.assert_array_equal(result[:,i],(raw-raw.mean())/raw.std())


def test_undefined_population_is_not_silently_repaired():
    v=np.ones((3,20),dtype=np.float64);start=np.zeros(3,dtype=np.int32);end=np.full(3,19,dtype=np.int32)
    assert np.isnan(blocks.correlations(v,start,end,[3],[1.],0)).all()
    assert blocks.popularity(pd.DataFrame(v)).isna().all()


@pytest.mark.parametrize('fault',['lag1','boolean_lag','negative_threshold','offset','shape','nan','bounds'])
def test_invalid_correlation_controls(fault):
    v=np.ones((3,20),dtype=np.float64);start=np.zeros(3,dtype=np.int32);end=np.full(3,19,dtype=np.int32)
    lags=[3];thresholds=[1.];offset=0
    if fault=='lag1':lags=[1]
    elif fault=='boolean_lag':lags=[True]
    elif fault=='negative_threshold':thresholds=[-1.]
    elif fault=='offset':offset=20
    elif fault=='shape':v=v.ravel()
    elif fault=='nan':v[0,0]=np.nan
    else:start[0]=20
    with pytest.raises(ValueError):blocks.correlations(v,start,end,lags,thresholds,offset)

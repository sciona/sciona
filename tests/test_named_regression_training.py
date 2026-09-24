import numpy as np
import pandas as pd
import pytest
from sciona.named_regression_training import temporal_regression_split, fit_catboost_regression


def split():
    frame = pd.DataFrame({'load':[1.,2.,3.,4.], 'machine':['A','B','A','B']},index=[5,5,8,2])
    times = pd.date_range('2031-01-01',periods=4,freq='h')
    return temporal_regression_split(frame,np.array([10.,20.,30.,40.]),times,
        np.array([True,False,True,True]),times[2],np.ones(4),['machine'])


def test_cutoff_eligibility_offsets_and_duplicate_indices():
    xt,yt,xv,yv,categories=split()
    assert xt.index.tolist()==[5] and xv.index.tolist()==[8,2] and categories==[1]
    np.testing.assert_array_equal(yt,[9.]);np.testing.assert_array_equal(yv,[29.,39.])


@pytest.mark.parametrize('parameters',[{'thread_count':4},{'task_type':'GPU'},{'train_dir':'unused'},{'allow_writing_files':True}])
def test_execution_policy_cannot_be_overridden(parameters):
    xt,yt,xv,yv,cat=split()
    with pytest.raises(ValueError):fit_catboost_regression(xt,yt,xv,yv,cat,parameters,2)


def test_small_native_fit_and_single_thread():
    x=pd.DataFrame({'load':np.arange(40,dtype=float),'machine':['A','B']*20})
    y=pd.Series(np.arange(40,dtype=float)*2)
    model=fit_catboost_regression(x.iloc[:30],y.iloc[:30],x.iloc[30:],y.iloc[30:],[1],
        dict(iterations=5,depth=2,random_seed=4,loss_function='MAE'),2)
    assert model.get_params()['thread_count']==1 and model.get_params()['allow_writing_files'] is False
    assert model.predict(x.iloc[30:],thread_count=1).shape==(10,)


def test_empty_temporal_partition_rejected():
    x=pd.DataFrame({'x':[1.]})
    with pytest.raises(ValueError,match='Nonempty eligible'):
        temporal_regression_split(x,np.ones(1),pd.DatetimeIndex(['2030-01-01']),np.ones(1,dtype=bool),
            '2031-01-01',np.zeros(1),[])

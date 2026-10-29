"""Synthetic exact-native comparisons for reusable configured fitting boundaries."""
import json
from unittest.mock import patch
import numpy as np
import pytest
import xgboost as xgb
from threadpoolctl import threadpool_info
from sciona.atoms.ml.xgboost.configured_model_io import (
    fit_configured_regression_model,fit_configured_binary_model,
    witness_fit_configured_regression_model,witness_fit_configured_binary_model,
)
from sciona.atoms.ml.xgboost.model_io import predict_regression_model,predict_binary_model,witness_predict_regression_model
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.flavours_classifiers import BRANCHES

REGRESSION=dict(eta=.05,max_depth=6,min_child_weight=10,subsample=.8,colsample_bytree=.9,seed=1,tree_method='exact',base_score=.5)
CONFIGURATIONS=[('regression',REGRESSION,2800)]+[
    ('binary',dict(eta=eta,max_depth=depth,min_child_weight=child,subsample=1.,
        colsample_bytree=columns,seed=seed,tree_method='exact',base_score=.5),rounds)
    for _,rounds,eta,depth,child,columns,seed in BRANCHES.values()]


def synthetic(domain):
    rng=np.random.default_rng(864)
    if domain=='oscillations':
        x=rng.normal(size=(80,3));query=rng.normal(size=(11,3))
        target=np.sin(x[:,0])+x[:,1]*x[:,2]
    else:
        x=rng.uniform(-2,3,size=(80,5));query=rng.uniform(-1,4,size=(11,5))
        target=3*x[:,0]**2-x[:,1]+.2*x[:,4]
    x[::13,1]=np.nan;query[::4,1]=np.nan
    return x,target,query


@pytest.mark.parametrize('domain',['oscillations','polynomial_response'])
@pytest.mark.parametrize('kind,configuration,rounds',CONFIGURATIONS)
def test_full_source_fit_controls_match_independent_native_predictions(domain,kind,configuration,rounds):
    features,target,query=synthetic(domain)
    if kind=='binary':target=(target>np.median(target)).astype(np.int64)
    names=['coordinate_'+str(i) for i in range(features.shape[1])]
    fit=fit_configured_regression_model if kind=='regression' else fit_configured_binary_model
    predict=predict_regression_model if kind=='regression' else predict_binary_model
    before=xgb.get_config()
    state,actual=fit(features,target,names,configuration,rounds)
    assert xgb.get_config()==before and actual==rounds
    parameters=dict(configuration,objective='reg:squarederror' if kind=='regression' else 'binary:logistic',nthread=1)
    with xgb.config_context(nthread=1):
        reference=xgb.train(parameters,xgb.DMatrix(features,label=target,missing=np.nan,nthread=1),num_boost_round=rounds)
        expected=reference.predict(xgb.DMatrix(query,missing=np.nan,nthread=1))
        observed=predict(query,names,json.loads(json.dumps(state)))
    np.testing.assert_array_equal(observed,expected)
    assert observed.shape==(11,) and np.isfinite(observed).all()


@pytest.mark.parametrize('key,value',[
    ('eta',0),('eta',True),('max_depth',0),('max_depth',2.5),('min_child_weight',-1),
    ('subsample',1.1),('colsample_bytree',0),('seed',True),('base_score',np.nan),
    ('tree_method','hist'),('nthread',4),('objective','reg:absoluteerror'),
])
def test_unsupported_configuration_rejected_before_fit(key,value):
    x,y,_=synthetic('oscillations');configuration=dict(REGRESSION);configuration[key]=value
    with patch.object(xgb,'train',side_effect=AssertionError('Invalid configuration reached training')):
        with pytest.raises(ValueError):fit_configured_regression_model(x,y,['a','b','c'],configuration,3)


@pytest.mark.parametrize('rounds',[0,-1,True,2.5])
def test_invalid_round_counts_fail_before_fit(rounds):
    x,y,_=synthetic('oscillations')
    with pytest.raises(ValueError):fit_configured_regression_model(x,y,['a','b','c'],REGRESSION,rounds)


def test_configuration_is_complete_and_binary_base_score_is_valid():
    x,y,_=synthetic('oscillations');labels=(y>np.median(y)).astype(np.int64)
    config=dict(REGRESSION);del config['seed']
    with pytest.raises(ValueError):fit_configured_regression_model(x,y,['a','b','c'],config,3)
    for value in [0,1,-1]:
        with pytest.raises(ValueError):fit_configured_binary_model(x,labels,['a','b','c'],dict(REGRESSION,base_score=value),3)


def test_runtime_inputs_and_feature_names_are_checked():
    x,y,_=synthetic('oscillations')
    for a,b,names in [(x.astype(np.float32),y,['a','b','c']), (x,y[:-1],['a','b','c']),
        (x,y,['a','a','c']), (np.full_like(x,np.inf),y,['a','b','c'])]:
        with pytest.raises(ValueError):fit_configured_regression_model(a,b,names,REGRESSION,3)
    with pytest.raises(ValueError):fit_configured_binary_model(x,np.zeros(len(y),dtype=np.int64),['a','b','c'],REGRESSION,3)


def test_witness_preserves_target_units_and_native_state_compatibility():
    units=DimensionalSignature(L=1)
    state,rounds=witness_fit_configured_regression_model(AbstractArray(shape=(80,3)),
        AbstractArray(shape=(80,),dim=units),['a','b','c'],REGRESSION,2800)
    result=witness_predict_regression_model(AbstractArray(shape=(11,3)),['a','b','c'],state)
    assert rounds==2800 and result.shape==(11,) and result.dim==units
    with pytest.raises(ValueError,match='dimensionless'):
        witness_fit_configured_binary_model(AbstractArray(shape=(80,3)),
            AbstractArray(shape=(80,),dtype='int64',dim=units),['a','b','c'],REGRESSION,3)


def test_native_scope_and_model_round_count_are_enforced():
    x,y,_=synthetic('oscillations');train=xgb.train;before=xgb.get_config()
    def checked(*args,**kwargs):
        assert xgb.get_config()==before and args[0]['nthread']==1
        assert all(pool['num_threads']==1 for pool in threadpool_info())
        return train(*args,**kwargs)
    with patch.object(xgb,'train',side_effect=checked):
        _,rounds=fit_configured_regression_model(x,y,['a','b','c'],REGRESSION,3)
    assert rounds==3 and xgb.get_config()==before
    class Incomplete:
        def num_boosted_rounds(self):return 2
    with patch.object(xgb,'train',return_value=Incomplete()):
        with pytest.raises(RuntimeError,match='complete requested'):
            fit_configured_regression_model(x,y,['a','b','c'],REGRESSION,3)
    assert xgb.get_config()==before

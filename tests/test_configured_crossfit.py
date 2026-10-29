"""Reusable fold selection and training isolation on synthetic numerical arrays."""
import numpy as np
import pytest
import xgboost as xgb
from sklearn.model_selection import KFold
from sciona.atoms.ml.model_selection.configured_crossfit import (
    shuffled_kfold_masks,fit_masked_configured_regression,fit_masked_configured_binary,
    witness_shuffled_kfold_masks,witness_fit_masked_configured_regression,
)
from sciona.atoms.ml.xgboost.model_io import predict_regression_model,predict_binary_model
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature

CONFIG=dict(eta=.05,max_depth=6,min_child_weight=10,subsample=.8,colsample_bytree=.9,seed=1,tree_method='exact',base_score=.5)


@pytest.mark.parametrize('seed',[1,2,3,4,5,6,555])
def test_all_source_fold_masks_match_backend_with_exact_row_coverage(seed):
    x=np.arange(92,dtype=np.float64).reshape(23,4)
    coverage=np.zeros(23,dtype=np.int64)
    for fold,(fit,held) in enumerate(KFold(5,shuffle=True,random_state=seed).split(x)):
        selected,excluded=shuffled_kfold_masks(x,5,seed,fold)
        np.testing.assert_array_equal(np.flatnonzero(selected),fit)
        np.testing.assert_array_equal(np.flatnonzero(excluded),held)
        assert selected.dtype==excluded.dtype==np.bool_
        assert selected.any() and excluded.any() and np.all(selected^excluded)
        coverage+=excluded
        changed=shuffled_kfold_masks(np.full_like(x,np.nan),5,seed,fold)
        np.testing.assert_array_equal(changed[0],selected)
        np.testing.assert_array_equal(changed[1],excluded)
    np.testing.assert_array_equal(coverage,np.ones(23,dtype=np.int64))


@pytest.mark.parametrize('kind,rounds',[('regression',2800),('binary',1500)])
def test_masked_full_budget_fits_match_native_and_ignore_unselected_values(kind,rounds):
    rng=np.random.default_rng(882)
    x=rng.normal(size=(80,4));target=x[:,0]+x[:,1]**2-x[:,2]
    if kind=='binary':target=(target>np.median(target)).astype(np.int64)
    query=rng.normal(size=(13,4));names=['a','b','c','d']
    selected,held=shuffled_kfold_masks(x,5,555,2)
    original_x=x.copy();original_target=target.copy();original_selection=selected.copy()
    fit=fit_masked_configured_regression if kind=='regression' else fit_masked_configured_binary
    predict=predict_regression_model if kind=='regression' else predict_binary_model
    parameters=dict(CONFIG)
    if kind=='binary':parameters.update(subsample=1.,colsample_bytree=1.)
    state,actual=fit(x,target,names,selected,parameters,rounds)
    changed_x=x.copy();changed_x[held]=np.inf
    changed_target=target.copy();changed_target[held]=np.inf if kind=='regression' else -77
    isolated,isolated_rounds=fit(changed_x,changed_target,names,selected,parameters,rounds)
    with xgb.config_context(nthread=1):
        reference=xgb.train(dict(parameters,nthread=1,objective='reg:squarederror' if kind=='regression' else 'binary:logistic'),
            xgb.DMatrix(x[selected],label=target[selected],nthread=1),num_boost_round=rounds)
        expected=reference.predict(xgb.DMatrix(query,nthread=1))
        observed=predict(query,names,state)
        np.testing.assert_array_equal(observed,expected)
        np.testing.assert_array_equal(predict(query,names,isolated),expected)
    assert actual==isolated_rounds==rounds
    np.testing.assert_array_equal(x,original_x)
    np.testing.assert_array_equal(target,original_target)
    np.testing.assert_array_equal(selected,original_selection)


@pytest.mark.parametrize('splits,seed,fold',[(1,1,0),(24,1,0),(5,-1,0),(5,2**32,0),(5,True,0),(5,1,-1),(5,1,5),(5,1,False),(5.5,1,0)])
def test_invalid_partition_controls_rejected(splits,seed,fold):
    with pytest.raises(ValueError):shuffled_kfold_masks(np.ones((23,4)),splits,seed,fold)


def test_invalid_mask_or_selected_class_population_rejected():
    x=np.arange(80,dtype=np.float64).reshape(20,4);y=np.arange(20,dtype=np.float64)
    for mask in [np.zeros(20,dtype=np.bool_),np.ones(19,dtype=np.bool_),np.ones(20,dtype=np.int64)]:
        with pytest.raises(ValueError):fit_masked_configured_regression(x,y,['a','b','c','d'],mask,CONFIG,3)
    mask=np.arange(20)<10
    labels=np.concatenate([np.zeros(10,dtype=np.int64),np.ones(10,dtype=np.int64)])
    with pytest.raises(ValueError,match='Both binary classes'):
        fit_masked_configured_binary(x,labels,['a','b','c','d'],mask,CONFIG,3)


def test_witnesses_preserve_target_units_and_validate_dimensionless_masks():
    features=AbstractArray(shape=(23,4));units=DimensionalSignature(L=1)
    fit,held=witness_shuffled_kfold_masks(features,5,555,2)
    assert fit.shape==held.shape==(23,) and fit.dtype==held.dtype=='bool'
    state,rounds=witness_fit_masked_configured_regression(features,AbstractArray(shape=(23,),dim=units),
        ['a','b','c','d'],fit,CONFIG,2800)
    assert state['target_dim']==units and rounds==2800
    with pytest.raises(ValueError,match='dimensionless'):
        witness_fit_masked_configured_regression(features,AbstractArray(shape=(23,),dim=units),
            ['a','b','c','d'],AbstractArray(shape=(23,),dtype='bool',dim=units),CONFIG,2800)

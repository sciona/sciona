"""Full-budget synthetic comparison of reusable fits with retained reconstruction.

This qualifies the 35-model numerical composition only. It does not qualify a
stored CDG, approve providers, or establish downstream nested validation.
"""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sciona.flavours_mass_correction import correct_mass
from sciona.atoms.ml.model_selection.configured_crossfit import (
    shuffled_kfold_masks, fit_masked_configured_regression,
)
from sciona.atoms.ml.xgboost.model_io import predict_regression_model


@pytest.mark.parametrize('rows,width,seed', [(63,3,812), (81,5,917)])
def test_all_reusable_mass_fits_match_retained_full_budget(rows,width,seed):
    rng=np.random.default_rng(seed)
    train=rng.normal(size=(rows,width))
    query=rng.normal(size=(11,width))
    target=2.0*train[:,0]+train[:,1]**2+0.1*rng.normal(size=rows)
    train[::13,-1]=np.nan
    query[::5,-1]=np.nan
    names=[f'feature_{i}' for i in range(width)]
    config=dict(eta=.05,max_depth=6,min_child_weight=10,subsample=.8,
                colsample_bytree=.9,seed=1,tree_method='exact',base_score=.5)
    sums=np.zeros(rows,dtype=np.float64)
    coverage=np.zeros(rows,dtype=np.int64)
    query_sum=np.zeros(len(query),dtype=np.float64)
    rounds=[]
    # Explicitly retain source fold order and float64 accumulation of native
    # float32 predictions. All training/query/control rows remain separate.
    with threadpool_limits(limits=1):
        reference=correct_mass(train,target,query)
        for fold_seed in (1,2,3,4,5,6,555):
            for fold in range(5):
                selected,held=shuffled_kfold_masks(train,5,fold_seed,fold)
                state,actual=fit_masked_configured_regression(
                    train,target,names,selected,config,2800)
                predictions=predict_regression_model(train,names,state)
                sums[held]+=predictions[held]
                coverage[held]+=1
                query_sum+=predict_regression_model(query,names,state)
                rounds.append(actual)
    assert rounds==[2800]*35
    np.testing.assert_array_equal(coverage,np.full(rows,7,dtype=np.int64))
    np.testing.assert_array_equal(coverage,reference['oof_coverage'])
    np.testing.assert_array_equal(sums/coverage,reference['oof_mass'])
    np.testing.assert_array_equal(query_sum/35,reference['query_mass'])
    assert tuple(rounds)==reference['fit_rounds']
    assert np.ptp(query_sum/35)>0

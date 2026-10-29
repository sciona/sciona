"""Selected scaler reuse, exact backend parity and portable state contracts."""
import json
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sciona.atoms.ml.model_selection.masked_standardization import (
    fit_masked_standardization,apply_standardization,
    witness_fit_masked_standardization,witness_apply_standardization,
)
from sciona.atoms.ml.model_selection.configured_crossfit import shuffled_kfold_masks
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature


@pytest.mark.parametrize('width,seed',[(3,421),(5,912)])
def test_all_outer_fold_scalers_match_backend_and_ignore_held_rows(width,seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(600,width));query=rng.normal(size=(17,width))
    x[:,-1]=7.;query[:,-1]=8.
    names=[f'feature_{i}' for i in range(width)]
    original=x.copy();original_query=query.copy()
    for fold in range(5):
        selected,held=shuffled_kfold_masks(x,5,555,fold)
        state=fit_masked_standardization(x,names,selected)
        reference=StandardScaler().fit(x[selected])
        np.testing.assert_array_equal(state['mean'],reference.mean_)
        np.testing.assert_array_equal(state['variance'],reference.var_)
        np.testing.assert_array_equal(state['scale'],reference.scale_)
        assert state['samples']==selected.sum() and state['scale'][-1]==1
        restored=json.loads(json.dumps(state))
        np.testing.assert_array_equal(apply_standardization(x,names,restored),reference.transform(x))
        np.testing.assert_array_equal(apply_standardization(query,names,restored),reference.transform(query))
        changed=x.copy();changed[held]=np.inf
        assert fit_masked_standardization(changed,names,selected)==state
    np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(query,original_query)


@pytest.mark.parametrize('fault',['names','width','scale','variance','samples','nonfinite'])
def test_invalid_numeric_states_rejected(fault):
    x=np.arange(24,dtype=np.float64).reshape(8,3);names=['a','b','c']
    state=fit_masked_standardization(x,names,np.ones(8,dtype=bool))
    if fault=='names':state['feature_names']=names[::-1]
    elif fault=='width':state['mean']=state['mean'][:-1]
    elif fault=='scale':state['scale'][0]=0
    elif fault=='variance':state['variance'][0]=-1
    elif fault=='samples':state['samples']=True
    else:state['mean'][0]=np.inf
    with pytest.raises(ValueError):apply_standardization(x,names,state)


def test_witnesses_track_input_units_and_emit_dimensionless_features():
    units=DimensionalSignature(L=1);x=AbstractArray(shape=(10,3),dim=units)
    state=witness_fit_masked_standardization(x,['a','b','c'],AbstractArray(shape=(10,),dtype='bool'))
    output=witness_apply_standardization(AbstractArray(shape=(4,3),dim=units),['a','b','c'],state)
    assert output.shape==(4,3) and output.dim.is_dimensionless
    with pytest.raises(ValueError,match='fitted units'):
        witness_apply_standardization(AbstractArray(shape=(4,3),dim=DimensionalSignature(T=1)),['a','b','c'],state)


def test_invalid_selection_and_nonfinite_selected_values_rejected():
    x=np.ones((8,3));names=['a','b','c']
    for mask in [np.zeros(8,dtype=bool),np.ones(7,dtype=bool),np.ones(8,dtype=np.int64)]:
        with pytest.raises(ValueError):fit_masked_standardization(x,names,mask)
    x[0,0]=np.nan
    with pytest.raises(ValueError):fit_masked_standardization(x,names,np.ones(8,dtype=bool))

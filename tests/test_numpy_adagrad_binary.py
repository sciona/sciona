"""Reusable neural fits against the retained full-epoch numerical reference."""
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sciona.flavours_neural import loss_gradients,predict
from sciona.atoms.ml.numpy_adagrad_binary import (
    fit_masked_adagrad_binary,predict_adagrad_binary,_loss_gradients,
    witness_fit_masked_adagrad_binary,witness_predict_adagrad_binary,
)
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature

CONFIG=dict(hidden_units=8,epochs=3000,batch_size=128,seed=0,learning_rate=.01,epsilon=1e-6)


@pytest.mark.parametrize('width,seed',[(3,0),(5,9)])
def test_full_3000_epoch_parameters_predictions_and_selected_training_isolation(width,seed):
    rng=np.random.default_rng(710+seed)
    x=rng.normal(size=(173,width));y=(x[:,0]+x[:,1]**2>.7).astype(np.int64)
    selection=np.arange(len(x))%7!=0
    query=rng.normal(size=(19,width));names=[f'feature_{i}' for i in range(width)]
    config=dict(CONFIG,seed=seed)
    state,epochs,updates=fit_masked_adagrad_binary(x,y,names,selection,config)
    changed=x.copy();changed[~selection]=np.inf
    changed_y=y.copy();changed_y[~selection]=-77
    isolated,isolated_epochs,isolated_updates=fit_masked_adagrad_binary(changed,changed_y,names,selection,config)
    assert isolated==state and isolated_epochs==epochs==3000
    assert updates==isolated_updates==6000
    init=np.random.RandomState(seed)
    params=[init.uniform(-np.sqrt(6/(width+8)),np.sqrt(6/(width+8)),(width,8)),
            np.zeros(8),init.uniform(-np.sqrt(.6),np.sqrt(.6),(8,2)),np.zeros(2)]
    accumulators=[np.zeros_like(p) for p in params];tx,ty=x[selection],y[selection]
    with threadpool_limits(limits=1):
        for _ in range(3000):
            for start in range(0,len(tx),128):
                _,gradients=loss_gradients(tx[start:start+128],ty[start:start+128],params)
                for p,accumulator,gradient in zip(params,accumulators,gradients):
                    accumulator+=gradient*gradient
                    p-=.01*gradient/np.sqrt(accumulator+1e-6)
        expected=predict(query,params)
    for actual,reference in zip(state['parameters'],params):np.testing.assert_array_equal(actual,reference)
    actual=predict_adagrad_binary(query,names,json.loads(json.dumps(state)))
    np.testing.assert_array_equal(actual,expected)
    assert np.ptp(actual)>0 and np.all((actual>=0)&(actual<=1))


def test_loss_gradient_matches_finite_differences_away_from_relu_kinks():
    rng=np.random.default_rng(513);x=rng.normal(size=(9,3));y=np.arange(9,dtype=np.int64)%2
    params=[rng.normal(size=(3,4)),np.full(4,3.),rng.normal(size=(4,2)),np.zeros(2)]
    _,gradients=_loss_gradients(x,y,params)
    for tensor,gradient in zip(params,gradients):
        for index in np.ndindex(tensor.shape):
            old=tensor[index];h=1e-6
            tensor[index]=old+h;plus=_loss_gradients(x,y,params)[0]
            tensor[index]=old-h;minus=_loss_gradients(x,y,params)[0]
            tensor[index]=old
            assert gradient[index]==pytest.approx((plus-minus)/(2*h),rel=2e-5,abs=2e-8)


@pytest.mark.parametrize('key,value',[('epochs',0),('batch_size',False),('hidden_units',0),('seed',-1),
    ('seed',2**32),('learning_rate',0),('epsilon',float('nan')),('epochs',3.5)])
def test_invalid_controls_rejected(key,value):
    with pytest.raises(ValueError):
        fit_masked_adagrad_binary(np.ones((4,2)),np.array([0,1,0,1]),['a','b'],np.ones(4,dtype=bool),dict(CONFIG,**{key:value}))


def test_state_names_shapes_and_nonfinite_parameters_rejected():
    x=np.array([[0.,1.],[1.,0.],[2.,1.],[1.,2.]])
    state,_,_=fit_masked_adagrad_binary(x,np.array([0,1,0,1]),['a','b'],np.ones(4,dtype=bool),dict(CONFIG,epochs=2))
    with pytest.raises(ValueError):predict_adagrad_binary(x,['b','a'],state)
    bad=json.loads(json.dumps(state));bad['parameters'][3]=[0.]
    with pytest.raises(ValueError):predict_adagrad_binary(x,['a','b'],bad)
    bad=json.loads(json.dumps(state));bad['parameters'][0][0][0]=np.inf
    with pytest.raises(ValueError):predict_adagrad_binary(x,['a','b'],bad)
    with pytest.raises(ValueError):fit_masked_adagrad_binary(x,np.zeros(4,dtype=np.int64),['a','b'],np.ones(4,dtype=bool),CONFIG)


def test_witness_dimension_and_model_contracts():
    features=AbstractArray(shape=(20,3));targets=AbstractArray(shape=(20,),dtype='int64')
    mask=AbstractArray(shape=(20,),dtype='bool')
    state,epochs,updates=witness_fit_masked_adagrad_binary(features,targets,['a','b','c'],mask,CONFIG)
    assert epochs==3000 and updates.min_val==3000
    assert witness_predict_adagrad_binary(AbstractArray(shape=(5,3)),['a','b','c'],state).shape==(5,)
    with pytest.raises(ValueError,match='dimensionless'):
        witness_fit_masked_adagrad_binary(AbstractArray(shape=(20,3),dim=DimensionalSignature(L=1)),targets,['a','b','c'],mask,CONFIG)

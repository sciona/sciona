"""Full 50-model numerical composition before neural CDG assembly."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from sciona.flavours_neural import train_neural,loss_gradients
from sciona.atoms.ml.model_selection.configured_crossfit import shuffled_kfold_masks
from sciona.atoms.ml.model_selection.nested_stratified_masks import stratified_inner_masks
from sciona.atoms.ml.model_selection.masked_standardization import fit_masked_standardization,apply_standardization
from sciona.atoms.ml.numpy_adagrad_binary import fit_masked_adagrad_binary,predict_adagrad_binary


def test_full_fifty_model_composition_matches_retained_neural_reference():
    rng=np.random.default_rng(641)
    features=rng.normal(size=(600,3));query=rng.normal(size=(17,3))
    labels=np.tile(np.array([0,1],dtype=np.int64),300);rng.shuffle(labels)
    features[:,0]+=labels*.7
    names=['a','b','c'];oof=np.zeros(600);prediction=np.zeros(17);fits=[]
    held_counts=np.zeros(600,dtype=np.int64)
    with threadpool_limits(limits=1):
        expected=train_neural(features,labels,query)
        for fold in range(5):
            outer,held=shuffled_kfold_masks(features,5,555,fold)
            inner,validation=stratified_inner_masks(labels,outer,200,0)
            scaler=fit_masked_standardization(features,names,outer)
            scaled=apply_standardization(features,names,json.loads(json.dumps(scaler)))
            scaled_query=apply_standardization(query,names,scaler)
            for seed in range(10):
                configuration=dict(hidden_units=8,epochs=3000,batch_size=128,seed=seed,learning_rate=.01,epsilon=1e-6)
                state,epochs,updates=fit_masked_adagrad_binary(scaled,labels,names,inner,configuration)
                restored=json.loads(json.dumps(state))
                hp=predict_adagrad_binary(scaled[held],names,restored)
                qp=predict_adagrad_binary(scaled_query,names,restored)
                # Source divides each contribution before ordered summation.
                # Replacing this with sum-then-divide changes rounding.
                oof[held]+=hp/10;prediction+=qp/50;held_counts[held]+=1
                params=[np.asarray(p) for p in state['parameters']]
                validation_loss,_=loss_gradients(scaled[validation],labels[validation],params)
                fits.append(dict(fold=fold,seed=seed,epochs=epochs,optimizer_updates=updates,
                                 validation_loss=float(validation_loss)))
    assert len(fits)==50 and fits==expected['fits']
    assert sum(f['epochs'] for f in fits)==150000
    assert sum(f['optimizer_updates'] for f in fits)==600000
    np.testing.assert_array_equal(held_counts,np.full(600,10,dtype=np.int64))
    np.testing.assert_array_equal(oof,expected['oof'])
    np.testing.assert_array_equal(prediction,expected['query'])
    assert np.ptp(oof)>0 and np.ptp(prediction)>0

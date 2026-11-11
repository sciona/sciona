import numpy as np
import pytest

from sciona.tgs_fit_execution import fit_arrays, predecessor_receipt


def test_final_torch_folds_require_pseudo_last_cycle_not_best():
    fit = dict(branch='pytorch', weight_predecessor='torch.p2.f0')
    with pytest.raises(ValueError, match='missing predecessor'):
        predecessor_receipt(fit, {})
    previous = dict(cycle_checkpoints={'50': 'first', '100': 'second', '150': 'final'}, best_checkpoint='wrong')
    assert predecessor_receipt(fit, {'torch.p2.f0': previous}) == 'final'
    del previous['cycle_checkpoints']['100']
    with pytest.raises(ValueError, match='complete pseudo-only'):
        predecessor_receipt(fit, {'torch.p2.f0': previous})


def test_keras_predecessor_uses_validation_best_not_periodic():
    fit = dict(branch='keras', weight_predecessor='keras.r1.p1.f0')
    assert predecessor_receipt(fit, {fit['weight_predecessor']: dict(best_checkpoint='best', periodic_checkpoints={40: 'periodic'})}) == 'best'


def test_materialized_pseudo_masks_preserve_keras_uint8_boundary():
    population = dict(labeled_images=np.zeros((10,101,101,3),dtype=np.uint8),
                      labeled_masks=np.ones((10,101,101),dtype=bool),
                      query_images=np.full((2,101,101,3),7,dtype=np.uint8),
                      folds=np.tile(np.arange(5),2), labeled_nonconstant=np.ones(10,dtype=bool),
                      query_nonconstant=np.ones(2,dtype=bool))
    pseudo = dict(stage=1,masks=np.ones((2,101,101),dtype=bool),confidence=np.ones(2),area=np.full(2,10201))
    fit = dict(key='keras.r2.p0.f0',branch='keras',pseudo_round=1)
    arrays, counts = fit_arrays(fit,population,pseudo)
    assert arrays[1].dtype == np.uint8 and (arrays[1] == 255).all()
    assert arrays[3].dtype == np.uint8 and (arrays[3] == 255).all()
    assert (arrays[0] == 7).all()
    assert counts == dict(training_labeled=0,training_query=2,validation_labeled=2)
    with pytest.raises(ValueError, match='round missing'):
        fit_arrays(fit,population,dict(pseudo,stage=2))

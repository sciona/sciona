"""Synthetic checks for reusable inner splits and source neural partitions."""
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from sciona.atoms.ml.model_selection.nested_stratified_masks import (
    stratified_inner_masks,witness_stratified_inner_masks,
)
from sciona.atoms.ml.model_selection.configured_crossfit import shuffled_kfold_masks
from sciona.flavours_neural import neural_partitions
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature


def test_all_five_source_outer_and_inner_masks_match_retained_neural_plan():
    rng=np.random.default_rng(921)
    labels=np.tile(np.array([0,1],dtype=np.int64),300)
    rng.shuffle(labels)
    features=rng.normal(size=(len(labels),3))
    reference=neural_partitions(labels)
    for fold,(fit,held,inner,validation) in enumerate(reference):
        outer,excluded=shuffled_kfold_masks(features,5,555,fold)
        inner_mask,validation_mask=stratified_inner_masks(labels,outer,200,0)
        np.testing.assert_array_equal(np.flatnonzero(outer),fit)
        np.testing.assert_array_equal(np.flatnonzero(excluded),held)
        np.testing.assert_array_equal(np.flatnonzero(inner_mask),fit[inner])
        np.testing.assert_array_equal(np.flatnonzero(validation_mask),fit[validation])
        assert not np.any(inner_mask & validation_mask)
        np.testing.assert_array_equal(inner_mask | validation_mask,outer)
        assert not np.any((inner_mask | validation_mask) & excluded)
        changed=labels.copy();changed[excluded]=-999
        observed=stratified_inner_masks(changed,outer,200,0)
        np.testing.assert_array_equal(observed[0],inner_mask)
        np.testing.assert_array_equal(observed[1],validation_mask)


def test_multiclass_noncontiguous_labels_all_folds_and_input_preservation():
    labels=np.tile(np.array([-7,2,31],dtype=np.int64),17)
    outer=np.arange(len(labels))%7!=0
    before_y=labels.copy();before_mask=outer.copy()
    indices=np.flatnonzero(outer);coverage=np.zeros(len(labels),dtype=np.int64)
    for fold,(fit,validation) in enumerate(StratifiedKFold(5).split(np.zeros((len(indices),1)),labels[outer])):
        actual_fit,actual_validation=stratified_inner_masks(labels,outer,5,fold)
        np.testing.assert_array_equal(np.flatnonzero(actual_fit),indices[fit])
        np.testing.assert_array_equal(np.flatnonzero(actual_validation),indices[validation])
        coverage+=actual_validation
    np.testing.assert_array_equal(coverage,outer.astype(np.int64))
    np.testing.assert_array_equal(labels,before_y)
    np.testing.assert_array_equal(outer,before_mask)


@pytest.mark.parametrize('splits,fold',[(1,0),(5,-1),(5,5),(True,0),(5,False),(5.5,0)])
def test_invalid_controls_rejected(splits,fold):
    labels=np.tile(np.array([0,1],dtype=np.int64),10)
    with pytest.raises(ValueError):stratified_inner_masks(labels,np.ones(20,dtype=bool),splits,fold)


@pytest.mark.parametrize('case',['one_class','too_few','empty','misaligned','nonboolean','float_labels'])
def test_invalid_selected_populations_rejected(case):
    labels=np.tile(np.array([0,1],dtype=np.int64),10);mask=np.ones(20,dtype=bool)
    if case=='one_class':mask=labels==0
    elif case=='too_few':mask[:15]=False
    elif case=='empty':mask[:]=False
    elif case=='misaligned':mask=mask[:-1]
    elif case=='nonboolean':mask=mask.astype(np.int64)
    else:labels=labels.astype(float)
    with pytest.raises(ValueError):stratified_inner_masks(labels,mask,5,0)


def test_witness_masks_and_dimension_guards():
    labels=AbstractArray(shape=(600,),dtype='int64');outer=AbstractArray(shape=(600,),dtype='bool')
    fit,validation=witness_stratified_inner_masks(labels,outer,200,0)
    assert fit.shape==validation.shape==(600,) and fit.dtype==validation.dtype=='bool'
    with pytest.raises(ValueError,match='dimensionless'):
        witness_stratified_inner_masks(AbstractArray(shape=(600,),dtype='int64',dim=DimensionalSignature(L=1)),outer,200,0)
    with pytest.raises(ValueError,match='dimensionless'):
        witness_stratified_inner_masks(labels,AbstractArray(shape=(600,),dtype='bool',dim=DimensionalSignature(T=1)),200,0)

"""Coverage, numerical order and unit contracts for reusable aggregation."""
import numpy as np
import pytest
from sciona.atoms.ml.model_selection.prediction_accumulation import (
    masked_prediction_contribution as contribution,
    merge_prediction_sums as merge,finalize_prediction_mean as finalize,
    witness_masked_prediction_contribution,witness_merge_prediction_sums,
    witness_finalize_prediction_mean,
)
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_selected_only_finiteness_and_no_mutation(dtype):
    values=np.array([3.,np.nan,np.inf,-2.],dtype=dtype);mask=np.array([True,False,False,True])
    total,counts=contribution(values,mask)
    assert total.dtype==np.float64
    np.testing.assert_array_equal(total,[3.,0.,0.,-2.])
    np.testing.assert_array_equal(counts,[1,0,0,1])
    total[0]=77;counts[0]=9
    assert values[0]==3 and mask[0]
    empty=contribution(values,np.zeros(4,dtype=bool))
    assert not empty[0].any() and not empty[1].any()
    with pytest.raises(ValueError):contribution(values,np.ones(4,dtype=bool))


def test_ordered_rounding_and_coverage_preserved():
    total=np.zeros(2);counts=np.zeros(2,dtype=np.int64)
    for value in (1e16,1.,-1e16):
        old=total.copy();old_counts=counts.copy()
        total,counts=merge(total,counts,np.full(2,value),np.ones(2,dtype=np.int64))
        assert not np.shares_memory(total,old)
        np.testing.assert_array_equal(counts,old_counts+1)
    np.testing.assert_array_equal(finalize(total,counts,3),[0.,0.])


@pytest.mark.parametrize('counts,expected', [([0,1],1),([1,2],1),([2,2],1),([1,1],0),([1,1],True),([1,1],1.5)])
def test_missing_excess_or_invalid_expected_coverage_rejected(counts,expected):
    with pytest.raises(ValueError):finalize(np.zeros(2),np.array(counts,dtype=np.int64),expected)


def test_invalid_states_alignment_and_overflow_rejected():
    ones=np.ones(2,dtype=np.int64)
    for total,counts in [(np.ones(2),np.zeros(2,dtype=np.int64)),
                         (np.zeros(2),-ones),(np.zeros(2),ones.astype(float)),
                         (np.zeros((2,1)),ones),(np.array([np.inf,0.]),ones)]:
        with pytest.raises(ValueError):merge(total,counts,np.zeros(2),ones)
    with pytest.raises(ValueError,match='align'):merge(np.zeros(2),ones,np.zeros(1),ones[:1])
    with pytest.raises(ValueError,match='count overflow'):
        merge(np.zeros(2),np.full(2,np.iinfo(np.int64).max),np.zeros(2),ones)
    with pytest.raises(ValueError,match='sum overflow'):
        merge(np.full(2,1e308),ones,np.full(2,1e308),ones)


def test_witnesses_preserve_units_and_reject_incompatible_units():
    units=DimensionalSignature(L=1)
    predictions=AbstractArray(shape=(3,),dim=units)
    mask=AbstractArray(shape=(3,),dtype='bool')
    total,counts=witness_masked_prediction_contribution(predictions,mask)
    combined,coverage=witness_merge_prediction_sums(total,counts,total,counts)
    assert witness_finalize_prediction_mean(combined,coverage,7).dim==units
    with pytest.raises(ValueError,match='compatible units'):
        witness_merge_prediction_sums(total,counts,AbstractArray(shape=(3,),dim=DimensionalSignature(T=1)),counts)
    with pytest.raises(ValueError,match='dimensionless'):
        witness_finalize_prediction_mean(total,AbstractArray(shape=(3,),dtype='int64',dim=units),7)
    with pytest.raises(ValueError,match='dimensionless'):
        witness_masked_prediction_contribution(predictions,AbstractArray(shape=(3,),dtype='bool',dim=units))

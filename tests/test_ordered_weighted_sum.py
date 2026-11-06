"""Synthetic cross-domain numerical contracts and source blend fidelity."""
import numpy as np
import pytest
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum as combine
from sciona.atoms.numerical.statistics.ordered_sum import witness_ordered_weighted_sum
from sciona.amex_pipeline import blend


@pytest.mark.parametrize('dtype',[np.int32,np.float32,np.float64])
@pytest.mark.parametrize('order',['C','F'])
def test_arbitrary_signed_non_normalized_coefficients(dtype,order):
    x=np.array([[2,5],[-3,7],[8,-2]],dtype=dtype,order=order)
    w=np.array([2.,-3.,0.]);original=x.copy();weights=w.copy()
    actual=combine(x,w)
    np.testing.assert_array_equal(actual,[13.,-11.])
    assert actual.dtype==np.float64 and not np.shares_memory(actual,x)
    np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(w,weights)


def test_order_sensitive_cancellation():
    # A reassociated sum can give zero instead of one.
    x=np.array([[1e16],[-1e16],[1.]])
    np.testing.assert_array_equal(combine(x,np.ones(3)),[1.])


def test_single_row_signed_zero_and_fresh_output():
    x=np.array([[-0.,3.]])
    result=combine(x,np.ones(1))
    assert np.signbit(result[0]) and not np.shares_memory(result,x)
    np.testing.assert_array_equal(result,[-0.,3.])


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('size',[1,8,103])
def test_source_recipe_exact_without_normalizing_weights(dtype,size):
    x=np.random.default_rng(212).uniform(size=(4,size)).astype(dtype)
    np.testing.assert_array_equal(combine(x,np.array([.3,.35,.15,.1])),blend(*x))
    # Preserve literal left-to-right floating arithmetic, including its last bit.
    np.testing.assert_array_equal(combine(np.ones((4,size)),np.array([.3,.35,.15,.1])),
                                  np.full(size,.8999999999999999))


@pytest.mark.parametrize('x,w',[
    ([[1.]],np.ones(1)),(np.ones(2),np.ones(2)),(np.ones((1,1,1)),np.ones(1)),
    (np.empty((0,2)),np.empty(0)),(np.empty((2,0)),np.ones(2)),
    (np.array([[True]]),np.ones(1)),(np.array([[1j]]),np.ones(1)),
    (np.array([[np.nan]]),np.ones(1)),(np.array([[np.inf]]),np.ones(1)),
    (np.ones((2,2)),[1.,1.]),(np.ones((2,2)),np.ones(1)),
    (np.ones((2,2)),np.ones((2,1))),(np.ones((1,2)),np.array([True])),
    (np.ones((1,2)),np.array([np.nan])),(np.ones((1,2)),np.array([np.inf])),
    (np.array([[1e308]]),np.array([2.])),
    (np.array([[1e308],[1e308],[-1e308]]),np.ones(3)),
])
def test_invalid_inputs_and_intermediate_overflow_rejected(x,w):
    with pytest.raises(ValueError):combine(x,w)


def test_symbolic_execution_not_claimed():
    with pytest.raises(ValueError):witness_ordered_weighted_sum({}, {})

import numpy as np
import pytest
from sciona.atoms.numerical.statistics.empirical_features import empirical_entropy,mixed_power_moment,stack_feature_columns,witness_stack_feature_columns
from sciona.atoms.causal_inference.feature_primitives.pair_orientation import orient_typed_pairs,augment_pair_labels,witness_orient_typed_pairs
from sciona.ghost.abstract import AbstractArray


def test_entropy_known_distribution_and_joint_symbol_order():
    samples=np.array([5,5,7,9])
    p=np.array([.5,.25,.25])
    expected=-sum(p*np.log(p))
    assert empirical_entropy(samples)==pytest.approx(expected)
    assert empirical_entropy(samples,.7)==pytest.approx(expected+.7*2/8)
    joint=np.column_stack([samples,[2,2,3,4]])
    assert empirical_entropy(joint,.7)==empirical_entropy(samples,.7)
    assert empirical_entropy(np.ones(8),.7)==0


def test_generic_entropy_and_moments_match_existing_recipe_definitions():
    from sciona.atoms.causal_inference.feature_primitives.selected_features import _count_entropy
    rng=np.random.default_rng(523)
    for n in [3,17,128]:
        a=rng.integers(0,5,n);b=rng.integers(0,3,n)
        assert empirical_entropy(a,.7)==_count_entropy(a)
        assert empirical_entropy(np.column_stack([a,b]),.7)==_count_entropy(list(zip(a,b)))
        x=rng.normal(size=n);y=rng.normal(size=n)
        for power in [2,3]:
            assert mixed_power_moment(x,y,power,1)==np.mean(x**power*y)


def test_integer_moment_uses_float_arithmetic_and_detects_overflow():
    assert mixed_power_moment(np.array([100000],dtype=np.int32),np.array([2]),2,1)==2e10
    with pytest.raises(FloatingPointError):mixed_power_moment(np.array([1e200]),np.array([1.]),2,1)


def test_stacking_is_ordered_and_has_independent_storage():
    columns=[np.array([2.,4.]),np.array([7.,9.]),np.array([3.,6.])]
    actual=stack_feature_columns(columns)
    np.testing.assert_array_equal(actual,[[2,7,3],[4,9,6]])
    actual[0,0]=100
    assert columns[0][0]==2
    metadata=witness_stack_feature_columns([AbstractArray(shape=(2,),dtype='float64')]*3)
    assert metadata.shape==(2,3)


def test_orientation_preserves_types_ragged_shapes_and_isolates_arrays():
    pairs=[(np.array([1.,2.,3.]),np.array([0,1,0])),(np.arange(5.),np.array([2,3,2,4,3]))]
    types=[('Numerical','Binary'),('Numerical','Categorical')]
    xs,xt,ys,yt=orient_typed_pairs(pairs,types)
    assert xt==['Numerical','Binary','Numerical','Categorical']
    assert yt==['Binary','Numerical','Categorical','Numerical']
    for i,(a,b) in enumerate(pairs):
        np.testing.assert_array_equal(xs[2*i],a);np.testing.assert_array_equal(ys[2*i],b)
        np.testing.assert_array_equal(xs[2*i+1],b);np.testing.assert_array_equal(ys[2*i+1],a)
    xs[0][0]=100
    assert pairs[0][0][0]==1 and ys[1][0]==1
    abstract=witness_orient_typed_pairs([(AbstractArray(shape=a.shape,dtype=str(a.dtype)),AbstractArray(shape=b.shape,dtype=str(b.dtype))) for a,b in pairs],types)
    assert [a.shape for a in abstract[0]]==[(3,),(3,),(5,),(5,)]


@pytest.mark.parametrize('dtype',[np.int8,np.int64,np.uint8,np.uint64])
def test_label_negation_is_signed_and_does_not_mutate(dtype):
    labels=np.array([1,0,1],dtype=dtype)
    result=augment_pair_labels(labels,[None]*3)
    np.testing.assert_array_equal(result,[1,-1,0,0,1,-1])
    assert result.dtype==np.int64
    np.testing.assert_array_equal(labels,[1,0,1])


@pytest.mark.parametrize('operation',[
    lambda:empirical_entropy(np.array([])),
    lambda:empirical_entropy(np.array([np.nan])),
    lambda:empirical_entropy(np.array([1,2]),-.1),
    lambda:empirical_entropy(np.array([1,2]),True),
    lambda:mixed_power_moment(np.ones(2),np.ones(3),2,1),
    lambda:mixed_power_moment(np.ones(2),np.ones(2),True,1),
    lambda:mixed_power_moment(np.ones(2),np.ones(2),-1,1),
    lambda:stack_feature_columns([np.ones(2),np.ones(3)]),
    lambda:stack_feature_columns([np.array([np.inf])]),
    lambda:orient_typed_pairs([(np.ones(3),np.arange(3))],[('Numerical','Numerical')]),
    lambda:orient_typed_pairs([(np.arange(3),np.arange(4))],[('Numerical','Numerical')]),
    lambda:orient_typed_pairs([(np.arange(3),np.arange(3))],[('Binary','Numerical')]),
    lambda:orient_typed_pairs([(np.arange(3),np.arange(3))],[('Unknown','Numerical')]),
    lambda:augment_pair_labels(np.array([2]),[None]),
    lambda:augment_pair_labels(np.array([1.]),[None]),
    lambda:augment_pair_labels(np.array([1]),[None,None]),
])
def test_invalid_inputs_rejected(operation):
    with pytest.raises(ValueError):operation()

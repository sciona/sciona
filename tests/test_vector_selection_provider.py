"""Synthetic vector selection checks beyond the source's three-dimensional use."""
from types import SimpleNamespace
import numpy as np
import pytest
from sciona.atoms.physics.vector_selection import select_by_absolute_inner_product as select
from sciona.trackml_cells import CellFeatures


@pytest.mark.parametrize('size',[0,1,7,31])
@pytest.mark.parametrize('dimension',[1,2,3,8])
def test_independent_choice_and_preservation(size,dimension):
    rng=np.random.default_rng(71)
    arrays=[rng.normal(size=(size,dimension)) for _ in range(3)]
    before=[a.copy() for a in arrays]
    chosen,inner=select(*arrays)
    for i,(ref,a,b) in enumerate(zip(*arrays)):
        left=sum(float(x)*float(y) for x,y in zip(ref,a))
        right=sum(float(x)*float(y) for x,y in zip(ref,b))
        expected=a if abs(left)>abs(right) else b
        score=left if abs(left)>abs(right) else right
        np.testing.assert_array_equal(chosen[i],expected*(-1 if score<0 else 1))
        np.testing.assert_allclose(inner[i],abs(score),rtol=1e-14,atol=1e-14)
    for a,b in zip(arrays,before):np.testing.assert_array_equal(a,b)
    assert chosen.shape==(size,dimension) and inner.shape==(size,)


@pytest.mark.parametrize('size',[0,1,7,31])
def test_exact_retained_cell_selection(size):
    rng=np.random.default_rng(11)
    ref,a,b=[rng.normal(size=(size,3)) for _ in range(3)]
    source=SimpleNamespace(_cell_features=(a,b,np.ones(size,dtype=np.int16)))
    expected=CellFeatures.estimateClosestDirection(source,np.arange(size),ref)
    actual=select(ref,a,b)
    for left,right in zip(actual,expected):np.testing.assert_array_equal(left,right)


def test_ties_zero_and_nonunit_candidates():
    ref=np.array([[1.,0.],[0.,0.],[1.,0.]])
    a=np.array([[1.,2.],[2.,3.],[1.,0.]])
    b=np.array([[-1.,4.],[5.,6.],[2.,3.]])
    vector,inner=select(ref,a,b)
    np.testing.assert_array_equal(vector,[[-1.,4.],[5.,6.],[2.,3.]]*np.array([[-1.],[1.],[1.]]))
    np.testing.assert_array_equal(inner,[1.,0.,2.])


@pytest.mark.parametrize('bad',[np.ones(3),np.ones((1,0)),np.ones((1,3),dtype=np.float32),np.full((1,3),np.nan)])
def test_bad_reference(bad):
    with pytest.raises(ValueError):select(bad,np.ones((1,3)),np.ones((1,3)))


def test_alignment_overflow_and_registered_witness():
    with pytest.raises(ValueError):select(np.ones((1,3)),np.ones((2,3)),np.ones((1,3)))
    with pytest.raises(ValueError,match='arithmetic'):
        select(np.full((1,3),1e308),np.full((1,3),1e308),np.zeros((1,3)))
    from sciona.ghost.registry import get_witness
    with pytest.raises(ValueError):get_witness(select.__module__+'.'+select.__name__)()

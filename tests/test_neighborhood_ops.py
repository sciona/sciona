"""Independent stochastic-matrix oracle and synthetic cross-domain invariants."""
import numpy as np
import pytest

from sciona.neighborhood_ops import blend_neighbor_means
from sciona.geotemporal_postprocess import smooth


@pytest.mark.parametrize('strength',[0.,.25,1.])
def test_explicit_matrix_oracle(strength):
    values = np.array([-4.,2.,8.,-10.])
    mask = np.array([[0,1,1,0],[1,0,0,0],[0,0,1,1],[0,0,0,0]],dtype=bool)
    means = np.array([[0,.5,.5,0],[1,0,0,0],[0,0,.5,.5],[0,0,0,1]])
    before = values.copy(),mask.copy()
    result = blend_neighbor_means(values,mask,strength)
    np.testing.assert_allclose(result,((1-strength)*np.eye(4)+strength*means)@values)
    np.testing.assert_array_equal(values,before[0]);np.testing.assert_array_equal(mask,before[1])
    assert result.dtype == np.float64 and not np.shares_memory(result,values)


def test_permutation_affine_equivariance_and_local_bounds():
    rng=np.random.default_rng(42)
    for n in (1,3,9):
        values=rng.normal(size=n);mask=rng.random((n,n))<.4
        order=rng.permutation(n)
        actual=blend_neighbor_means(values,mask,.3)
        permuted=blend_neighbor_means(values[order],mask[np.ix_(order,order)],.3)
        np.testing.assert_allclose(permuted,actual[order])
        np.testing.assert_allclose(blend_neighbor_means(-2*values+7,mask,.3),-2*actual+7)
        for i,row in enumerate(mask):
            local=np.r_[values[i],values[row]]
            assert local.min()-1e-12 <= actual[i] <= local.max()+1e-12


def test_empty_neighborhoods_and_finite_extremes():
    values=np.array([np.finfo(float).max,np.finfo(float).max])
    np.testing.assert_array_equal(blend_neighbor_means(values,np.ones((2,2),dtype=bool),.5),values)
    np.testing.assert_array_equal(blend_neighbor_means([-2,8],np.zeros((2,2),dtype=bool),1),[-2,8])


def test_directed_neighborhood_does_not_claim_mass_conservation():
    result=blend_neighbor_means([0,10],[[False,True],[False,True]],1)
    np.testing.assert_array_equal(result,[10,10])
    assert result.sum()!=10


def test_geographic_clipping_and_adjacency_are_explicit_recipe_policy():
    points=[dict(x=0.,y=0.,time=1.),dict(x=1.,y=0.,time=1.),dict(x=0.,y=0.,time=2.)]
    mask=np.array([[False,True,False],[True,False,False],[False,False,False]])
    values=np.array([-2.,4.,100.])
    result=blend_neighbor_means(np.maximum(values,0),mask,.25)
    np.testing.assert_allclose(result,smooth(points,values,radius=2.,strength=.25))
    assert blend_neighbor_means(values,mask,0)[0] == -2


@pytest.mark.parametrize('values,mask,strength',[
    ([],[],.5),([[1,2]],[[True]],.5),([True],[[True]],.5),
    (['2'],[[True]],.5),([1j],[[True]],.5),([np.nan],[[True]],.5),
    ([np.inf],[[True]],.5),([1,2],[[True]],.5),([1],[[1]],.5),
    ([1],[[True]],True),([1],[[True]],-.1),([1],[[True]],1.1),
    ([1],[[True]],float('nan')),([1],[[True]],float('inf')),([1],[[True]],10**1000),
])
def test_reject_ambiguous_or_nonfinite_contract(values,mask,strength):
    with pytest.raises(ValueError):blend_neighbor_means(values,mask,strength)

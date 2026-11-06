"""Contract checks for the source-preserving direction injection boundary."""
import numpy as np
import pytest
from sciona.decomposed_trackml_direction_lifecycle import direction
from sciona.atoms.particle_tracking.helix_geometry.atoms import helix_direction_from_two_points as provider


def inputs():
    return [np.array([v],dtype=np.float64) for v in (0,0,2,1,0,0,0,1,0)]


def test_equal_height_and_empty_boundary():
    np.testing.assert_array_equal(direction(*inputs(),operation=provider), [2.])
    def forbidden(*args):
        raise AssertionError('Empty internal batches bypass nonempty provider')
    result=direction(*[np.empty(0) for _ in range(9)],operation=forbidden)
    assert result.dtype==np.float64 and result.shape==(0,)


@pytest.mark.parametrize('bad',[np.ones(2),np.ones((1,1)),np.ones(1,dtype=np.float32),np.array([np.nan]),np.array([np.inf])])
def test_invalid_input_rejects_before_provider(bad):
    args=inputs();args[3]=bad
    def forbidden(*args):raise AssertionError('Provider must not receive invalid inputs')
    with pytest.raises(ValueError):direction(*args,operation=forbidden)


def test_hidden_cross_product_overflow_rejects_before_provider():
    args=inputs();args[3]=np.array([1e308]);args[7]=np.array([1e308])
    def forbidden(*args):raise AssertionError('Overflow must reject before provider')
    with pytest.raises(ValueError,match='arithmetic'):direction(*args,operation=forbidden)


@pytest.mark.parametrize('bad',[np.array([np.inf]),np.array([1.],dtype=np.float32),np.ones(2),1.])
def test_invalid_provider_output_rejects(bad):
    with pytest.raises(ValueError):direction(*inputs(),operation=lambda *args:bad)


def test_noncallable_rejects():
    with pytest.raises(ValueError):direction(*inputs(),operation=None)

import numpy as np
import pytest
from sciona.atoms.numerical.spatial.volume_tiles import (
    split_volume_tiles,combine_volume_tiles,witness_split_volume_tiles,witness_combine_volume_tiles,
)
from sciona.ghost.abstract import AbstractArray


@pytest.mark.parametrize('shape',[(1,1,3,5),(2,7,9,11),(1,8,8,8)])
@pytest.mark.parametrize('margin',[0,2])
def test_roundtrip_order_edge_padding_and_channels(shape,margin):
    volume=np.arange(np.prod(shape),dtype=np.int32).reshape(shape)
    before=volume.copy()
    tiles,grid=split_volume_tiles(volume,4,2,margin)
    # Move channels to trailing payload axis for the spatial reconstruction.
    result=combine_volume_tiles(np.moveaxis(tiles,1,-1),grid,4,1,margin)
    expected=np.moveaxis(np.pad(volume,[(0,0)]+[(0,g*4-n) for g,n in zip(grid,shape[1:])],mode='edge'),0,-1)
    np.testing.assert_array_equal(result,expected)
    np.testing.assert_array_equal(volume,before)
    assert result.dtype==np.float32 and tiles.dtype==volume.dtype
    assert not np.shares_memory(tiles,volume)


@pytest.mark.parametrize('payload',[(),(2,),(2,3),(1,2,3)])
def test_payload_axes_are_opaque_and_stride_explicit(payload):
    values=np.arange(2*4**3*np.prod(payload or (1,)),dtype=float).reshape((2,4,4,4)+payload)
    result=combine_volume_tiles(values,(1,1,2),4,2,2)
    expected=np.concatenate([v[1:3,1:3,1:3] for v in values],axis=2).astype(np.float32)
    np.testing.assert_array_equal(result,expected)
    abstract=witness_combine_volume_tiles(AbstractArray(shape=values.shape,dtype='float64'),(1,1,2),4,2,2)
    assert abstract.shape==result.shape and abstract.dtype=='float32'


@pytest.mark.parametrize('side,stride,margin',[(0,1,0),(4,0,0),(4,2,-1),(4,2,4),(5,2,0),(4,2,1),(True,1,0)])
def test_bad_geometry_rejected(side,stride,margin):
    with pytest.raises(ValueError):split_volume_tiles(np.ones((1,3,3,3)),side,stride,margin)
    with pytest.raises(ValueError):combine_volume_tiles(np.ones((1,4,4,4)),(1,1,1),side,stride,margin)


def test_missing_tiles_and_nonfinite_values_rejected():
    with pytest.raises(ValueError):combine_volume_tiles(np.ones((1,4,4,4)),(1,1,2),4,1,0)
    with pytest.raises(ValueError):combine_volume_tiles(np.ones((2,4,4,4)),(1,1,1),4,1,0)
    with pytest.raises(ValueError):split_volume_tiles(np.full((1,2,2,2),np.nan),4,2,0)
    with pytest.raises(ValueError):combine_volume_tiles(np.full((1,4,4,4),np.inf),(1,1,1),4,1,0)
    with pytest.raises((ValueError,FloatingPointError)):
        combine_volume_tiles(np.full((1,4,4,4),1e100),(1,1,1),4,1,0)


def test_split_witness_matches_concrete_geometry():
    abstract,grid=witness_split_volume_tiles(AbstractArray(shape=(2,7,9,11),dtype='float32'),4,2,2)
    tiles,actual_grid=split_volume_tiles(np.zeros((2,7,9,11),dtype=np.float32),4,2,2)
    assert abstract.shape==tiles.shape and grid==actual_grid

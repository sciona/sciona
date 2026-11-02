import numpy as np
import pytest
from sciona.atoms.numerical.spatial.coordinate_scaling import scale_coordinate_rows, witness_scale_coordinate_rows
from sciona.ghost.abstract import AbstractArray


def test_signed_conversion_input_preservation_and_witness():
    rows = np.array([[2,4],[6,8]])
    result = scale_coordinate_rows(rows, [3,-2], [2,4], [1,-3])
    np.testing.assert_array_equal(result, [[2,1],[8,-1]])
    np.testing.assert_array_equal(rows, [[2,4],[6,8]])
    assert result.dtype == np.float64 and not np.shares_memory(rows,result)
    abstract = witness_scale_coordinate_rows(AbstractArray(shape=(2,2),dtype='int64'),[3,-2],[2,4],[1,-3])
    assert abstract.shape==(2,2) and abstract.dtype=='float64'


def test_empty_rows():
    assert scale_coordinate_rows(np.empty((0,3)),[1]*3,[1]*3,[0]*3).shape==(0,3)


@pytest.mark.parametrize('rows,n,d,o', [([[1]], [1],[0],[0]), ([[1]], [np.inf],[1],[0]),
    ([[1]], [1,2],[1],[0]), ([[np.nan]], [1],[1],[0]), ([[1e308]], [1e308],[1],[0]),
    ([1,2],[1],[1],[0]), ([[1j]], [1],[1],[0])])
def test_invalid_contract(rows,n,d,o):
    with pytest.raises(ValueError):scale_coordinate_rows(rows,n,d,o)

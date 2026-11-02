import pytest
import torch
from sciona.ghost.abstract import AbstractArray
from sciona.atoms.dl.spatial.centered_window import centered_window, witness_centered_window


def test_arbitrary_axes_alignment_view_and_gradients():
    values = torch.arange(5*6*7, dtype=torch.float64).reshape(5,6,7).requires_grad_()
    actual = centered_window(values, [2,0], [2,3])
    expected = values[1:4,:,2:4]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    actual.sum().backward()
    expected_gradient = torch.zeros_like(values)
    expected_gradient[1:4,:,2:4] = 1
    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)
    witness = witness_centered_window(AbstractArray(shape=(5,6,7),dtype='float64'), [2,0], [2,3])
    assert witness.shape == actual.shape and witness.dtype == 'float64'


@pytest.mark.parametrize('axes,widths', [([],[]),([0],[0]),([0],[6]),([0,0],[1,1]),([-1],[1]),([True],[1]),([0],[True]),([3],[1]),([0,1],[1])])
def test_invalid_windows(axes, widths):
    with pytest.raises(ValueError):
        centered_window(torch.ones(5,6,7), axes, widths)

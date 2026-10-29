"""Recipe-policy and full-gradient parity on synthetic detector predictions."""
from unittest.mock import patch
import pytest
import torch
from sciona import dsb_losses as original
from sciona import dsb_decomposed_losses as candidate


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('num_hard', [0, 1, 2, 20])
@pytest.mark.parametrize('positives', [True, False])
def test_detector_selection_policy_and_gradients(training, num_hard, positives):
    output = torch.linspace(-2, 2, 60, dtype=torch.float64).reshape(2, 6, 5).requires_grad_()
    reference = output.detach().clone().requires_grad_()
    labels = torch.zeros_like(output)
    labels[:, :, 0] = -1
    labels[:, 1, 0] = 0
    if positives:
        labels[:, 0, 0] = 1
    with patch.object(candidate, 'highest_scoring_indices', wraps=candidate.highest_scoring_indices) as selector:
        actual = candidate.detector_loss(output, labels, num_hard, training)
    expected = original.detector_loss(reference, labels, num_hard, training)
    assert selector.call_count == int(training and num_hard > 0)
    if selector.called:
        assert selector.call_args.args[2] == num_hard * output.shape[0]
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]
    actual_gradient, = torch.autograd.grad(actual['total'], output)
    expected_gradient, = torch.autograd.grad(expected['total'], reference)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

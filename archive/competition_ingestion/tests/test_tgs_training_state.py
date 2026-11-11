import math
import torch
import pytest

from sciona.tgs_training_state import optimizer_for, snapshot_learning_rate, capture_training_state, restore_training_state


def test_rmsprop_matches_two_step_scalar_formula():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = optimizer_for(model, 'keras', .01)
    expected, accumulator = 1., 0.
    for gradient in (2., -3.):
        model.weight.grad = torch.full_like(model.weight, gradient)
        optimizer.step()
        accumulator = .9 * accumulator + .1 * gradient ** 2
        expected -= .01 * gradient / (math.sqrt(accumulator) + 1e-7)
        assert model.weight.item() == pytest.approx(expected, abs=1e-7)


def test_cosine_restart_and_last_epoch():
    assert snapshot_learning_rate(0, 40, 1, .0001) == .0001
    assert snapshot_learning_rate(20, 40, 2, .0001) == .0001
    assert snapshot_learning_rate(39, 40, 1, .0001) == pytest.approx(.00005 * (1 + math.cos(39 * math.pi / 40)))


@pytest.mark.parametrize('branch', ['keras', 'pytorch'])
def test_snapshot_isolation_and_exact_optimizer_rng_resume(branch):
    torch.manual_seed(937)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 1))
    optimizer = optimizer_for(model, branch, .01)
    def step():
        optimizer.zero_grad()
        loss = model(torch.rand(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
    step()
    snapshot = capture_training_state(model, optimizer)
    original = {k: v.clone() for k, v in snapshot['model'].items()}
    step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    assert any(not torch.equal(expected[k], original[k]) for k in expected)
    for k in original:
        torch.testing.assert_close(snapshot['model'][k], original[k], rtol=0, atol=0)
    restore_training_state(model, optimizer, snapshot)
    step()
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)

"""Synthetic scalar objectives and finite differences for both Keras paths."""
import numpy as np
import pytest
import torch

from sciona.cassava_focal import focal_loss


def scalar(row, target, branch):
    p = np.exp(row - max(row))
    p /= sum(p)
    value = 0.
    for z, probability, label in zip(row, p, target):
        y = .9 * label + .02
        if branch == 'cached_logits':
            ce = np.logaddexp(0., z) - y * z
        else:
            clipped = min(max(probability, 1e-7), 1 - 1e-7)
            ce = -y * np.log(clipped + 1e-7) - (1 - y) * np.log(1 - clipped + 1e-7)
        pt = y * probability + (1 - y) * (1 - probability)
        value += (.25 * y + .75 * (1 - y)) * (1 - pt) ** 2 * ce
    return value


@pytest.mark.parametrize('branch', ['cached_logits', 'probabilities'])
def test_scalar_objective_gradient_and_immutability(branch):
    rng = np.random.default_rng(201)
    rows = rng.normal(size=(8, 5)) * 2
    targets = rng.dirichlet(np.ones(5), size=8)
    x = torch.tensor(rows, requires_grad=True)
    y = torch.tensor(targets)
    actual = focal_loss(x, y, bce_branch=branch)
    np.testing.assert_allclose(actual.detach(), [scalar(r, t, branch) for r, t in zip(rows, targets)], atol=1e-12)
    actual.sum().backward()
    expected = np.empty_like(rows)
    for i in range(8):
        for j in range(5):
            plus, minus = rows[i].copy(), rows[i].copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            expected[i, j] = (scalar(plus, targets[i], branch) - scalar(minus, targets[i], branch)) / 2e-5
    np.testing.assert_allclose(x.grad, expected, atol=1e-8)
    np.testing.assert_array_equal(y, targets)


def test_shift_distinguishes_cached_logits_from_probability_path():
    x = torch.tensor([[1., -2., 3., 0., -1.]], dtype=torch.float64)
    y = torch.tensor([[0., 0., 1., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(focal_loss(x, y, bce_branch='probabilities'), focal_loss(x + 10, y, bce_branch='probabilities'))
    assert not torch.allclose(focal_loss(x, y, bce_branch='cached_logits'), focal_loss(x + 10, y, bce_branch='cached_logits'))


@pytest.mark.parametrize('branch', ['cached_logits', 'probabilities'])
def test_extreme_finite_logits(branch):
    x = torch.tensor([[1000., -1000., 0., 1., -1.]], requires_grad=True)
    y = torch.tensor([[0., 1., 0., 0., 0.]])
    loss = focal_loss(x, y, bce_branch=branch)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(x.grad).all()


def test_branch_is_required():
    x = torch.zeros(1, 5)
    with pytest.raises(TypeError): focal_loss(x, torch.full_like(x, .2))
    with pytest.raises(ValueError): focal_loss(x, torch.full_like(x, .2), bce_branch='auto')


@pytest.mark.parametrize('kind', ['shape', 'nan', 'negative', 'sum', 'gradient'])
def test_invalid_targets(kind):
    x = torch.zeros(1, 5)
    y = torch.full_like(x, .2)
    if kind == 'shape': y = y[:, :4]
    if kind == 'nan': y[0, 0] = float('nan')
    if kind == 'negative': y[0, 0] = -.2
    if kind == 'sum': y *= .5
    if kind == 'gradient': y.requires_grad_()
    with pytest.raises(ValueError): focal_loss(x, y, bce_branch='cached_logits')

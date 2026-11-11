"""Synthetic tests for source accumulation and epoch scheduling semantics."""
import copy

import pytest
import torch

from sciona.cassava_loss import vit_loss
from sciona.cassava_vit_epoch import train_epoch


def test_two_batch_sum_tail_flush_and_scheduler_order():
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 5)
    oracle = copy.deepcopy(model)
    batches = [(torch.randn(n, 3), torch.arange(n) % 5) for n in (3, 2, 1)]
    opt = torch.optim.SGD(model.parameters(), lr=.2)
    reference = torch.optim.SGD(oracle.parameters(), lr=.2)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.1)
    result = train_epoch(model, batches, opt, schedule)
    for group in (batches[:2], batches[2:]):
        losses = [vit_loss(oracle(x), torch.nn.functional.one_hot(y, 5).float()).mean() for x, y in group]
        sum(losses).backward()
        reference.step()
        reference.zero_grad()
    for actual, expected in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(actual, expected)
    assert result == {'batches': 3, 'optimizer_steps': 2}
    assert opt.param_groups[0]['lr'] == pytest.approx(.02)


def test_invalid_late_batch_fails_before_mutation():
    model = torch.nn.Linear(3, 5)
    before = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters())
    good = (torch.zeros(2, 3), torch.zeros(2, dtype=torch.int64))
    bad = (torch.zeros(2, 3), torch.full((2,), 5, dtype=torch.int64))
    with pytest.raises(ValueError): train_epoch(model, [good, bad], opt)
    for key, value in model.state_dict().items(): torch.testing.assert_close(value, before[key])
    assert not opt.state

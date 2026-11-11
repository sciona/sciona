import json
from pathlib import Path

import pytest

from sciona.tgs_schedule import next_action


def plan():
    return json.loads((Path(__file__).resolve().parents[1] / 'docs/reviews/competition_tgs_training_plan.json').read_text())


def test_full_inventory_schedules_all_fits_and_both_pseudo_barriers():
    reviewed = plan()
    fits, rounds, actions = set(), set(), []
    while True:
        action = next_action(reviewed, fits, rounds)
        if action['kind'] == 'complete':
            break
        assert len(actions) < 66
        actions.append(action)
        if action['kind'] == 'fit':
            fits.add(action['key'])
        else:
            rounds.add(action['stage'])
    assert len(actions) == 66 and len(fits) == 63 and rounds == {1, 2, 3}
    order = {a.get('key', f"round{a.get('stage')}"): i for i, a in enumerate(actions)}
    assert order['torch.p0.f4'] < order['round1'] < order['keras.r2.p0.f0']
    assert order['torch.p1.f4'] < order['round2'] < order['torch.p2.f0']
    assert order['torch.p2.f0'] < order['torch.p3.f0'] < order['round3']


def test_cannot_infer_completed_fit_or_round_dependencies():
    with pytest.raises(ValueError, match='dependencies'):
        next_action(plan(), {'keras.r1.p1.f0'}, set())
    with pytest.raises(ValueError, match='dependencies'):
        next_action(plan(), set(), {1})


def test_rejects_unknown_and_truncated_inventory():
    with pytest.raises(ValueError, match='unknown'):
        next_action(plan(), {'invented'}, set())
    reviewed = plan()
    reviewed['fits'] = reviewed['fits'][:5]
    with pytest.raises(ValueError, match='63-fit'):
        next_action(reviewed, set(), set())

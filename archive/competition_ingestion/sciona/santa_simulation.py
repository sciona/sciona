"""Synthetic two-player bandit validation with hidden simulator thresholds.

The environment owns its thresholds and random draws; an agent receives only
its own reward history and public actions. This module makes no strength claim.
"""
import numpy as np
from sciona.santa_agent_features import BanditHistory


class BanditEnvironment:
    def __init__(self, thresholds):
        if type(thresholds) is not list or len(thresholds) != 100 or any(type(t) is not int or not 0 <= t <= 100 for t in thresholds):
            raise ValueError('One hundred integer thresholds required')
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.rounds = 0

    def step(self, actions, draws):
        if self.rounds >= 2000:
            raise ValueError('Episode is complete')
        for values, upper in [(actions,99),(draws,100)]:
            if type(values) is not list or len(values) != 2 or any(type(v) is not int or not 0 <= v <= upper for v in values):
                raise ValueError('Invalid two-player actions or integer draws')
        current = self.thresholds[actions]
        rewards = (np.asarray(draws) < current).astype(int).tolist()
        expected = (np.ceil(current) / 101.).tolist()
        counts = np.bincount(actions, minlength=100)
        self.thresholds *= np.power(.97, counts)
        self.rounds += 1
        return rewards, expected


def simulate(models, thresholds, *, rounds=2000, seed=42):
    if type(rounds) is not int or not 1 <= rounds <= 2000:
        raise ValueError('Round count must be in [1,2000]')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Explicit unsigned seed required')
    environment = BanditEnvironment(thresholds)
    history = BanditHistory()
    streams = np.random.SeedSequence(seed).spawn(3)
    decisions, opponent, draws = [np.random.default_rng(s) for s in streams]
    total = np.zeros(2, dtype=np.int64)
    expected_total = np.zeros(2, dtype=np.float64)
    for _ in range(rounds):
        decision = models.predict(history, seed=int(decisions.integers(0,2**32)))
        # A fixed random opponent is a correctness fixture, not a competitive
        # benchmark or a substitute for the winner's tournament evaluation.
        actions = [decision['action'], int(opponent.integers(0,100))]
        rewards, expected = environment.step(actions, draws.integers(0,101,size=2).tolist())
        history.observe(actions[0], actions[1], rewards[0])
        total += rewards
        expected_total += expected
    return dict(rounds=rounds, rewards=total.tolist(), expected_rewards=expected_total.tolist(),
                agent_observations=history.step, total_selections=int((history.own+history.opponent).sum()))

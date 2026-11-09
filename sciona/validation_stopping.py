"""Metric-independent, strict-improvement validation stopping policy.

Call once per completed evaluation. Checkpoint storage and metric computation
belong to the caller. Patience counts consecutive evaluations without an
improvement larger than min_delta; ties retain the earlier checkpoint.
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class ValidationStopping:
    patience: int
    min_delta: float = 0.
    mode: str = 'min'
    best: float | None = None
    best_index: int | None = None
    evaluations: int = 0
    stale: int = 0

    def __post_init__(self):
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError('Positive integer patience required')
        if type(self.min_delta) not in (int, float) or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('Finite nonnegative improvement threshold required')
        if self.mode not in ('min', 'max'):
            raise ValueError('Metric direction must be min or max')

    @property
    def stopped(self):
        return self.stale >= self.patience

    def observe(self, value):
        if self.stopped:
            raise ValueError('Stopping policy already exhausted')
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Finite metric required')
        improved = self.best is None or (
            value < self.best - self.min_delta if self.mode == 'min'
            else value > self.best + self.min_delta)
        return replace(self, best=value if improved else self.best,
                       best_index=self.evaluations if improved else self.best_index,
                       evaluations=self.evaluations + 1,
                       stale=0 if improved else self.stale + 1)

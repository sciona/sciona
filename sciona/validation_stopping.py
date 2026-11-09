"""Metric-independent, strict-improvement validation stopping policy.

Call once per completed evaluation. Checkpoint storage and metric computation
belong to the caller. Patience counts consecutive evaluations without an
improvement larger than min_delta; ties retain the earlier checkpoint.
"""
from dataclasses import dataclass, replace
import math


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


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
        if not finite_number(self.min_delta) or self.min_delta < 0:
            raise ValueError('Finite nonnegative improvement threshold required')
        if self.mode not in ('min', 'max'):
            raise ValueError('Metric direction must be min or max')
        if any(type(v) is not int or v < 0 for v in (self.evaluations, self.stale)) or self.stale > self.patience:
            raise ValueError('Invalid evaluation counters')
        if self.evaluations == 0:
            valid = self.best is None and self.best_index is None and self.stale == 0
        else:
            valid = (finite_number(self.best) and type(self.best_index) is int and
                     0 <= self.best_index < self.evaluations and
                     self.stale == self.evaluations - self.best_index - 1)
        if not valid:
            raise ValueError('Inconsistent checkpoint state')

    @property
    def stopped(self):
        return self.stale >= self.patience

    def observe(self, value):
        if self.stopped:
            raise ValueError('Stopping policy already exhausted')
        if not finite_number(value):
            raise ValueError('Finite metric required')
        improved = self.best is None or (
            value < self.best - self.min_delta if self.mode == 'min'
            else value > self.best + self.min_delta)
        return replace(self, best=value if improved else self.best,
                       best_index=self.evaluations if improved else self.best_index,
                       evaluations=self.evaluations + 1,
                       stale=0 if improved else self.stale + 1)

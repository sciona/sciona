"""TGS maximizing-metric callback control for the reconstructed Keras phases.

Matches configured EarlyStopping then ReduceLROnPlateau then best-checkpoint
ordering. Reference comparison uses Keras 2.2.4; winner version remains unproven.
"""
import math


class ValidationControl:
    def __init__(self, *, learning_rate, stop_patience, reduce_patience, factor, minimum):
        for value in (stop_patience, reduce_patience):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('nonnegative integer patience required')
        if not all(math.isfinite(v) for v in (learning_rate, factor, minimum)) or not 0 < factor < 1 or not 0 <= minimum <= learning_rate or learning_rate <= 0:
            raise ValueError('invalid learning-rate controls')
        self.rate, self.minimum, self.factor = learning_rate, minimum, factor
        self.stop_patience, self.reduce_patience = stop_patience, reduce_patience
        self.best = self.plateau_best = -math.inf
        self.stop_wait = self.reduce_wait = 0
        self.stopped = False
        self.epochs = 0

    def observe(self, score):
        if self.stopped:
            raise ValueError('training phase already stopped')
        if not math.isfinite(score):
            raise ValueError('finite validation metric required')
        improved = score > self.best
        if improved:
            self.best, self.stop_wait = score, 0
        else:
            self.stop_wait += 1
            self.stopped = self.stop_wait >= self.stop_patience
        # Source ReduceLROnPlateau has its own best and default min_delta=1e-4.
        reduced = False
        if score > self.plateau_best + 1e-4:
            self.plateau_best, self.reduce_wait = score, 0
        else:
            self.reduce_wait += 1
            if self.reduce_wait >= self.reduce_patience and self.rate > self.minimum:
                self.rate = max(self.minimum, self.rate * self.factor)
                self.reduce_wait = 0
                reduced = True
        self.epochs += 1
        return dict(epoch=self.epochs - 1, save_best=improved, stop=self.stopped,
                    learning_rate=self.rate, reduced=reduced)

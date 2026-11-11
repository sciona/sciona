"""Strict source checkpoint selection and stopping for 100-epoch base fits."""
import math


class BaseTrainingControl:
    def __init__(self, detector):
        if detector not in ('effdet', 'fasterrcnn'):
            raise ValueError('source detector family required')
        self.detector = detector
        self.best = math.inf if detector == 'effdet' else 0.0
        self.best_epoch = None
        self.patience = 0
        self.next_epoch = 0
        self.stopped = False

    def observe(self, epoch, metric):
        if self.stopped or type(epoch) is not int or epoch != self.next_epoch:
            raise ValueError('sequential unfinished source epoch required')
        if not math.isfinite(metric) or metric < 0:
            raise ValueError('finite nonnegative validation metric required')
        improved = metric < self.best if self.detector == 'effdet' else metric > self.best
        if improved:
            self.best, self.best_epoch, self.patience = float(metric), epoch, 0
        else:
            self.patience += 1
        self.stopped = self.patience == 40 or epoch == 99
        self.next_epoch += 1
        return dict(save_checkpoint=improved, stop=self.stopped, best_epoch=self.best_epoch,
                    best_metric=self.best, patience=self.patience)

    def selected_epoch(self):
        if not self.stopped:
            raise ValueError('source training budget or stopping condition not reached')
        if self.best_epoch is None:
            raise ValueError('source fit produced no qualifying checkpoint')
        return self.best_epoch

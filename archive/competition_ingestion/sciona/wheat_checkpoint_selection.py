"""Strict source validation selection for the two Global Wheat pseudo rounds.

Caller owns tensor snapshots and must retain the qualified round-one inference
checkpoint for round two. This tracks selection, not checkpoint provenance.
"""
import math


class PseudoCheckpointSelection:
    def __init__(self, *, round_number, inherited_best_loss=None):
        if type(round_number) is not int or round_number not in (1, 2):
            raise ValueError('pseudo round must be1or2')
        if round_number == 1:
            if inherited_best_loss is not None:
                raise ValueError('first pseudo round starts without an inherited validation best')
            best = math.inf
        else:
            self._validate_loss(inherited_best_loss)
            best = float(inherited_best_loss)
        self.round_number = round_number
        self.epochs = 10 if round_number == 1 else 6
        self.best_loss = best
        self.selected_epoch = None
        self.completed_epochs = 0

    @staticmethod
    def _validate_loss(loss):
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError('finite nonnegative validation loss required')

    def observe(self, epoch, validation_loss):
        if type(epoch) is not int or epoch != self.completed_epochs or epoch >= self.epochs:
            raise ValueError('sequential source pseudo epochs required')
        self._validate_loss(validation_loss)
        save = validation_loss < self.best_loss
        if save:
            self.best_loss = float(validation_loss)
            self.selected_epoch = epoch
        self.completed_epochs += 1
        return save

    def result(self):
        if self.completed_epochs != self.epochs:
            raise ValueError('pseudo validation epoch budget incomplete')
        inherited = self.selected_epoch is None
        if inherited and self.round_number != 2:
            raise ValueError('first pseudo round has no selected checkpoint')
        return dict(round_number=self.round_number, completed_epochs=self.completed_epochs,
                    selected_epoch=self.selected_epoch, selected_validation_loss=self.best_loss,
                    selected_origin='inherited_round1' if inherited else 'current_round')

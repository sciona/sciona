"""Keep recipe mutation and reservation policy around pure resource selection."""
import numpy as np

from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_bayesian_lifecycle import execute as bayesian_execute
from sciona.decomposed_trackml_correction_lifecycle import lifecycle_class
from sciona.trackml_candidates import Candidates


def commit_operation(function):
    if not callable(function):
        raise ValueError('Callable resource selection provider required')

    def commit(matrix, used, min_nhits=3, max_nloss=None, max_loss_fraction=1.,
               max_nrows=None, reserve_skipped=False):
        if (matrix.ndim != 2 or matrix.dtype.kind not in 'iu' or used.ndim != 1 or
                used.dtype != np.bool_ or not used.size or np.any(matrix < 0) or
                np.any(matrix >= used.size)):
            raise ValueError('Resource IDs must index a nonempty boolean mask')
        if np.any(np.count_nonzero(matrix, axis=1) == 0):
            raise ValueError('Source selection requires nonempty rows')
        loss = matrix.shape[1] if max_nloss is None else max_nloss
        limit = matrix.shape[0] if max_nrows is None else max_nrows
        for value, lower in ((min_nhits, 1), (loss, 0), (limit, 0)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError('Selection counts must be valid integers')
        if not np.isfinite(max_loss_fraction) or not 0 <= max_loss_fraction <= 1:
            raise ValueError('Loss fraction must lie in [0, 1]')
        ids = matrix.astype(np.int64, copy=True)
        if not reserve_skipped:
            selected, claimed = function(ids, used, min_nhits=min_nhits, max_nloss=loss,
                                         max_loss_fraction=max_loss_fraction, max_nrows=limit)
        else:
            # Reservation belongs to the recipe. Each row is decided against the
            # current mask; every visited resource is reserved for later rows.
            selected, claimed = np.zeros_like(ids), used.copy()
            claimed[0] = True
            accepted = 0
            for index, row in enumerate(ids):
                output, _ = function(row[None, :], claimed, min_nhits=min_nhits,
                    max_nloss=loss, max_loss_fraction=max_loss_fraction,
                    max_nrows=int(accepted < limit))
                selected[index] = output[0]
                accepted += int(np.any(output))
                claimed[row] = True
        matrix[...] = selected
        used[...] = claimed
    return commit


def candidate_class(function):
    class BoundCandidates(Candidates):
        pass
    BoundCandidates.submit = _bound(Candidates.submit, {'zeroUsedHits': commit_operation(function)})
    BoundCandidates.copy = _bound(Candidates.copy, {'Candidates': BoundCandidates})
    return BoundCandidates


def execute(*args, commit_function, **kwargs):
    candidates = candidate_class(commit_function)
    def local_class(*providers):
        base = lifecycle_class(*providers)
        class Run(base.Run):
            pass
        Run.__init__ = _bound(base.Run.__init__, {'Candidates': candidates})
        class Lifecycle(base):
            pass
        Lifecycle.Run = Run
        Lifecycle.findTracks = _bound(base.findTracks, {'Candidates': candidates})
        return Lifecycle
    return _bound(bayesian_execute, {'lifecycle_class': local_class})(*args, **kwargs)

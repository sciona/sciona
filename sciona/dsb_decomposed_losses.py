"""DSB loss recipe composed with the reusable thresholded log penalty.

Source arithmetic: lfz/DSB2017; MIT notice: docs/licenses/DSB2017-MIT.txt.
"""
import torch
from torch.nn import functional as F
from sciona.dsb_losses import _probabilities
from sciona.atoms.dl.loss.thresholded_log import thresholded_log_penalty


def classifier_loss(case_probability, proposal_probability, case_labels, known_proposals,
                    miss_threshold=.03, miss_ratio=1.):
    """Source case BCE plus batch/proposal-averaged log(p+0.001) miss penalty."""
    _probabilities(proposal_probability)
    batch, count = proposal_probability.shape
    for values, shape in [(case_probability, (batch,)), (case_labels, (batch,)),
                          (known_proposals, (batch, count))]:
        if (not isinstance(values, torch.Tensor) or values.shape != shape
                or values.dtype != proposal_probability.dtype or values.device != proposal_probability.device
                or not bool(torch.isfinite(values).all()) or not bool(((values >= 0) & (values <= 1)).all())):
            raise ValueError('classifier probabilities, labels, and proposal indicators must have matching contracts')
    if not 0 < miss_threshold <= 1 or not 0 <= miss_ratio < float('inf'):
        raise ValueError('invalid miss threshold or weight')
    classification = F.binary_cross_entropy(case_probability, case_labels)
    miss = thresholded_log_penalty(proposal_probability, known_proposals, miss_threshold, .001)
    return dict(total=classification + miss_ratio * miss, classification=classification, miss=miss)

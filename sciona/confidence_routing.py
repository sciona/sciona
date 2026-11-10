"""Reusable confidence routing and explicit weighted-utility calibration.

Domain adapters determine alternate-route eligibility and each route's utility
on independent validation examples. Utilities must share a higher-is-better
scale. This core infers no class vocabulary, validation mixture, metric, or
historical threshold grid. Calibration is not an unbiased performance estimate.
Numeric vectors are converted to float64 before comparison and scoring. An
adapter reproducing lower-precision comparisons must qualify threshold-boundary
rounding separately; existing domain routing implementations are not replaced.
"""
import numpy as np


def _vector(value, name, length=None):
    array = np.asarray(value)
    if (array.ndim != 1 or not len(array) or array.dtype.kind not in 'fiu'
            or not np.isfinite(array).all() or (length is not None and len(array) != length)):
        raise ValueError('Finite aligned numeric vector required: ' + name)
    return array.astype(np.float64)


def _inputs(confidence, alternate_eligible):
    confidence = _vector(confidence, 'confidence')
    eligible = np.asarray(alternate_eligible)
    if (confidence < 0).any() or (confidence > 1).any():
        raise ValueError('Confidence must lie in [0,1]')
    if eligible.dtype != np.bool_ or eligible.shape != confidence.shape:
        raise ValueError('Aligned boolean alternate eligibility required')
    return confidence, eligible


def confidence_route_mask(confidence, alternate_eligible, *, threshold):
    """Select alternate only below threshold and where it is eligible.

    Threshold equality retains the primary route. Eligibility encodes any
    domain fallback rule; this function does not infer it from predictions.
    """
    confidence, eligible = _inputs(confidence, alternate_eligible)
    candidate = np.asarray(threshold)
    if (candidate.ndim != 0 or candidate.dtype.kind not in 'fiu'
            or not np.isfinite(candidate) or not 0 <= float(candidate) <= 1):
        raise ValueError('Finite scalar threshold in [0,1] required')
    return (confidence < float(candidate)) & eligible


def fit_confidence_router(confidence, alternate_eligible, primary_utility, alternate_utility,
                          *, sample_weight, thresholds):
    """Maximize weighted utility on the explicit grid; ties choose smallest threshold.

    All candidate predictions/utilities and weights must come from a validation
    adapter. A supplied grid need not cover every distinct routing partition;
    the reported optimum is only over that grid. Zero-weight rows are ignored.
    """
    confidence, eligible = _inputs(confidence, alternate_eligible)
    primary = _vector(primary_utility, 'primary utility', len(confidence))
    alternate = _vector(alternate_utility, 'alternate utility', len(confidence))
    weights = _vector(sample_weight, 'sample weight', len(confidence))
    candidates = _vector(thresholds, 'thresholds')
    if (weights < 0).any() or weights.max() == 0:
        raise ValueError('Nonnegative weights with positive total required')
    if (candidates < 0).any() or (candidates > 1).any():
        raise ValueError('Candidate thresholds must lie in [0,1]')
    candidates = np.unique(candidates)
    # Scale before summing so a finite weight vector cannot overflow its total.
    weights = weights / weights.max()
    weights = weights / weights.sum()
    scores = []
    for threshold in candidates:
        use_alternate = (confidence < threshold) & eligible
        utility = np.where(use_alternate, alternate, primary)
        score = float(np.dot(weights, utility))
        if not np.isfinite(score):
            raise ValueError('Weighted utility is not representable as a finite float64')
        scores.append(score)
    best = int(np.argmax(scores))
    return dict(threshold=float(candidates[best]), weighted_utility=scores[best],
                candidate_thresholds=candidates, candidate_utilities=np.asarray(scores))

"""Explicit weighted component-recall calibration for grapheme routing.

This adapter owns 168/11/8 joint decoding, the submission consonant correction,
and 2:1:1 component weighting. Callers own validation independence, known-class
membership, mixture weights and threshold candidates. No historical fitted
threshold or mixture is inferred.
"""
import numpy as np

from sciona.bengali_joint_labels import decode_joint_classes
from sciona.component_recall import component_recall_utilities
from sciona.confidence_routing import fit_confidence_router


def submission_components(joint_classes):
    components = decode_joint_classes(joint_classes)
    components[:, 2] = np.where(components[:, 2] == 7, 2, components[:, 2])
    return components


def calibrate_grapheme_router(confidence, seen_predictions, unseen_predictions, truth_components,
                              *, seen_class_mask, sample_weight, thresholds):
    truth = np.asarray(truth_components)
    seen = submission_components(seen_predictions)
    unseen = submission_components(unseen_predictions)
    if (truth.shape != seen.shape or unseen.shape != seen.shape or truth.dtype.kind not in 'iu'
            or (truth < 0).any() or (truth >= np.array([168, 11, 7])).any()):
        raise ValueError('Aligned canonical submission truth components required')
    membership = np.asarray(seen_class_mask)
    if membership.shape != (14784,) or membership.dtype != np.bool_:
        raise ValueError('Explicit boolean joint-class membership map required')
    utilities = [component_recall_utilities(truth, prediction,
                 sample_weight=sample_weight, component_weight=[2., 1., 1.])
                 for prediction in (seen, unseen)]
    return fit_confidence_router(confidence, ~membership[np.asarray(unseen_predictions)], *utilities,
                                 sample_weight=np.ones(len(truth)), thresholds=thresholds)

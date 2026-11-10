"""Explicit B7 logit interpretation for the reconstructed routing boundary.

OOD confidence is the largest independent sigmoid score (paper Figure 2).
The seen branch uses softmax then joint-class argmax as an explicit inference
policy. The available reduced submission notebook omits this branch, so this
policy is not an attestation of the winner's exact historical decoder. The
two-font branch has its own separately recovered marginal decoder.
"""
import torch


def classifier_predictions(ood_logits, seen_logits):
    for logits, classes in ((ood_logits, 1295), (seen_logits, 14784)):
        if (not isinstance(logits, torch.Tensor) or logits.device.type != 'cpu'
                or logits.dtype != torch.float32 or logits.ndim != 2
                or logits.shape[0] < 1 or logits.shape[1] != classes
                or not torch.isfinite(logits).all()):
            raise ValueError('Finite branch-sized float32 CPU classifier logits required')
    if len(ood_logits) != len(seen_logits):
        raise ValueError('Aligned OOD and seen classifier rows required')
    with torch.no_grad():
        confidence = ood_logits.sigmoid().amax(dim=1)
        seen = seen_logits.softmax(dim=1).argmax(dim=1)
    # Float64 comparison is the explicit contract of the reusable router.
    return dict(ood_confidence=confidence.double().numpy(), seen_joint_classes=seen.numpy())

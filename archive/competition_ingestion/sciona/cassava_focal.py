"""EfficientNet focal objective with an explicit unresolved Keras BCE branch.

The winner's model returns softmax probabilities. Historical Keras BCE may
recover their cached pre-softmax logits. Callers must specify the established
runtime branch; neither branch here qualifies the original runtime by itself.
"""
import torch
import torch.nn.functional as F


def focal_loss(logits, targets, *, bce_branch):
    """Per-row five-class loss, alpha=.25, gamma=2, uniform smoothing=.1.

``cached_logits`` reproduces BCE from the pre-softmax logits while the focal
factor still uses softmax. ``probabilities`` uses historical Keras clipping and
epsilon inside both logarithms. Fixed targets only; no implicit branch choice.
"""
    if bce_branch not in ('cached_logits', 'probabilities'):
        raise ValueError('An explicit verified BCE branch is required')
    if (not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != 5
            or targets.shape != logits.shape or logits.dtype not in (torch.float32, torch.float64)
            or targets.dtype != logits.dtype or targets.device != logits.device or targets.requires_grad):
        raise ValueError('Aligned floating N-by-5 logits and fixed targets required')
    if (not torch.isfinite(logits).all() or not torch.isfinite(targets).all()
            or (targets < 0).any() or (targets > 1).any()
            or not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]), atol=1e-7, rtol=1e-6)):
        raise ValueError('Finite logits and normalized probability targets required')
    y = targets * .9 + .02
    p = logits.softmax(-1)
    if bce_branch == 'cached_logits':
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    else:
        clipped = p.clamp(1e-7, 1 - 1e-7)
        ce = -y * (clipped + 1e-7).log() - (1 - y) * (1 - clipped + 1e-7).log()
    pt = y * p + (1 - y) * (1 - p)
    alpha = y * .25 + (1 - y) * .75
    return (alpha * (1 - pt).square() * ce).sum(-1)

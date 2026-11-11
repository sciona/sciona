"""TGS branch-specific CPU losses; independent mathematics with hinge reuse.

Source method pinned in competition_tgs_source_triage.json. No winning code
vendored. Existing HuBMAP ordinary binary hinge is reused without modification.
"""
import torch
from torch.nn import functional as F

from sciona.hubmap_losses import _mask_pair, binary_lovasz_hinge


def keras_mask_pair(prediction, targets):
    """Keras resizing retains fractional masks; PyTorch branch remains binary."""
    for value in (prediction, targets):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
            raise ValueError('finite float32 CPU tensors required')
    if prediction.shape != targets.shape or prediction.ndim != 4 or prediction.shape[1] != 1 or not prediction.numel():
        raise ValueError('aligned nonempty N1HW tensors required')
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError('target coverage must be in [0,1]')


def keras_elu_lovasz(logits, targets):
    """Source ELU(error)+1 arithmetic, including fractional resized targets.

    Fractional-target use follows the source pipeline's arithmetic; no claim
    that its original binary-loss interpretation remains valid is made.
    """
    keras_mask_pair(logits, targets)
    terms = []
    for prediction, target in zip(logits.flatten(1), targets.flatten(1)):
        margins = 1 - prediction * (2 * target - 1)
        ordered, permutation = margins.sort(descending=True)
        foreground = target[permutation]
        intersection = foreground.sum() - foreground.cumsum(0)
        union = foreground.sum() + (1 - foreground).cumsum(0)
        cumulative = 1 - intersection / union
        increments = torch.diff(cumulative, prepend=cumulative.new_zeros(1)).detach()
        terms.append(torch.sum((F.elu(ordered) + 1) * increments))
    return torch.stack(terms).mean()


def keras_bce_dice(probabilities, targets, epsilon=1e-7):
    """Equal BCE/Dice mixture, global batch Dice and unit smoothing.

    Explicit clipping epsilon qualifies the modern probability-input runtime;
    exact historical TensorFlow floating-point operation parity is unproven.
    """
    keras_mask_pair(probabilities, targets)
    if not 0 < epsilon < 0.5:
        raise ValueError('epsilon must be in (0,0.5)')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('probabilities must lie in [0,1]')
    clipped = probabilities.clamp(epsilon, 1 - epsilon)
    bce = -(targets * clipped.log() + (1 - targets) * torch.log1p(-clipped)).mean()
    dice = (2 * (targets * probabilities).sum() + 1) / (targets.sum() + probabilities.sum() + 1)
    return (bce + 1 - dice) / 2


def pytorch_training_loss(logits, targets, *, pixel_logits=None, image_probabilities=None, image_targets=None):
    """Ordinary hinge, optionally plus source image-BCE and masked pixel hinge."""
    _mask_pair(logits, targets)
    supplied = [pixel_logits is not None, image_probabilities is not None, image_targets is not None]
    if any(supplied) and not all(supplied):
        raise ValueError('all three auxiliary inputs are required together')
    loss = binary_lovasz_hinge(logits, targets)
    if not any(supplied):
        return loss
    _mask_pair(pixel_logits, targets)
    for value in (image_probabilities, image_targets):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
            raise ValueError('finite float32 CPU image tensors required')
        if value.shape != (len(targets),):
            raise ValueError('image tensors must be aligned vectors')
    if not ((image_targets == 0) | (image_targets == 1)).all():
        raise ValueError('binary image targets required')
    if ((image_probabilities < 0) | (image_probabilities > 1)).any():
        raise ValueError('image probabilities must lie in [0,1]')
    nonempty = targets.flatten(1).sum(1) > 0
    # Keep an explicit zero gradient connection for empty auxiliary images.
    auxiliary = pixel_logits.sum() * 0
    if nonempty.any():
        auxiliary = auxiliary + binary_lovasz_hinge(pixel_logits[nonempty], targets[nonempty]) * nonempty.sum() / len(targets)
    return loss + F.binary_cross_entropy(image_probabilities, image_targets) + auxiliary

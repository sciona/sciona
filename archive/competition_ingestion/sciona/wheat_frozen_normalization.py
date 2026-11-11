"""Frozen normalization matching the winner's torchvision 0.5 backbone."""
from torchvision.ops.misc import FrozenBatchNorm2d


def source_frozen_batch_norm(channels: int) -> FrozenBatchNorm2d:
    """Preserve the historical variance formula when building ResNet-152."""
    return FrozenBatchNorm2d(channels, eps=0.0)
